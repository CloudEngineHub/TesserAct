import os
import cv2
import torch
from PIL import Image
import numpy as np
from diffusers import CogVideoXImageToVideoPipeline, CogVideoXDPMScheduler
from diffusers.utils import load_image, export_to_video
from tesseract.utils import crop_and_resize_frames, print_memory
from huggingface_hub import snapshot_download

torch.set_grad_enabled(False)
# seed everything
seed = 42
torch.manual_seed(seed)
np.random.seed(seed)


def validate(
    pretrained_model_path: str,
    lora_weights_path: str,
    validation_prompts: list,
    validation_images: list,
    output_dir: str,
    guidance_scale: float = 6.0,
    use_dynamic_cfg: bool = True,
    height: int = 256,
    width: int = 320,
    num_validation_videos: int = 1,
    fps: int = 8,
    mixed_precision: str = "bf16",
):
    """
    Simplified validation process:
    1. Load the pretrained pipeline
    2. Load LoRA weights
    3. For each validation prompt and image pair, generate video(s) and save them

    Args:
        pretrained_model_path (str): Path or identifier to the pretrained model, e.g., "THUDM/CogVideoX-5b-I2V".
        lora_weights_path (str): Path to the trained LoRA weights.
        validation_prompts (list): A list of prompt strings used for validation.
        validation_images (list): A list of image paths/URLs corresponding to the validation prompts.
        output_dir (str): Directory to save the generated validation videos.
        guidance_scale (float): Guidance scale for generation.
        use_dynamic_cfg (bool): Whether to use dynamic CFG during generation.
        height (int): Generated video height.
        width (int): Generated video width.
        num_validation_videos (int): Number of videos to generate per prompt.
        fps (int): Frame rate of the output video.
        mixed_precision (str): Mixed precision setting, one of "fp16", "bf16", or "no".
    """
    # Determine the weight dtype
    if mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif mixed_precision == "bf16":
        weight_dtype = torch.bfloat16
    else:
        weight_dtype = torch.float32

    # Load the pipeline
    pipe = CogVideoXImageToVideoPipeline.from_pretrained(pretrained_model_path, torch_dtype=weight_dtype).to("cuda")
    del pipe.transformer.patch_embed.pos_embedding
    pipe.transformer.patch_embed.use_learned_positional_embeddings = False
    pipe.transformer.config.use_learned_positional_embeddings = False

    # Load DPMScheduler
    pipe.scheduler = CogVideoXDPMScheduler.from_config(pipe.scheduler.config)

    # Load LoRA weights
    if lora_weights_path.count("/") == 2:
        list_of_files = lora_weights_path.split("/")
        repo_id = "/".join(list_of_files[:2])  # Get namespace/repo_name
        subfolder = list_of_files[2]  # Get the subfolder
        lora_weights_path = snapshot_download(
            repo_id=repo_id,
            local_dir_use_symlinks=False,
        )
        lora_weights_path = os.path.join(lora_weights_path, subfolder)

    pipe.load_lora_weights(lora_weights_path, adapter_name="cogvideox-lora")

    # Set LoRA adapter scaling factor (adjust as needed)
    pipe.set_adapters(["cogvideox-lora"], [1.0])

    # Run validation
    # for prompt, image_path in zip(validation_prompts, validation_images):
    for im_idx, (prompt, image_path) in enumerate(zip(validation_prompts, validation_images)):
        print(f"Running validation for prompt: {prompt}")
        print(f"Using image: {image_path}")
        image = load_image(image_path)
        image = crop_and_resize_frames([np.array(image)], (height, width))[0]
        image = Image.fromarray(image)
        image = torch.from_numpy(np.array(image)).to("cuda") / 255.0
        image = image.permute(2, 0, 1).unsqueeze(0)

        for idx in range(num_validation_videos):
            result = pipe(
                image=image,
                prompt=prompt,
                guidance_scale=guidance_scale,
                use_dynamic_cfg=use_dynamic_cfg,
                height=height,
                width=width,
                num_inference_steps=50,  # Adjust inference steps as needed
            )
            print_memory("cuda")
            video_frames = result.frames[0]

            # Save the video
            safe_prompt = prompt.replace(" ", "_")
            output_file = os.path.join(output_dir, f"val_{im_idx}_{safe_prompt}_{idx}.mp4")
            export_to_video(video_frames, output_file, fps=fps)
            print(f"Saved validation video to: {output_file}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--weights_path", type=str, default="anyezhy/tesseract/tesseract_v01e_rgb_lora")
    parser.add_argument("--image_path", type=str, default="asset/images/fruit_vangogh.png")
    parser.add_argument("--prompt", type=str, default="pick up the apple google robot")
    parser.add_argument("--memory_efficient", action="store_true", default=False)
    args = parser.parse_args()

    pretrained_model = "THUDM/CogVideoX-5b-I2V"  # always use this model
    lora_weights = args.weights_path

    val_images = [args.image_path]
    val_prompts = [args.prompt]
    out_dir = "./results"
    os.makedirs(out_dir, exist_ok=True)

    validate(
        pretrained_model_path=pretrained_model,
        lora_weights_path=lora_weights,
        validation_prompts=val_prompts,
        validation_images=val_images,
        output_dir=out_dir,
        guidance_scale=7.5,
        use_dynamic_cfg=True,
        height=480,
        width=640,
        num_validation_videos=1,
        fps=8,
        mixed_precision="fp16",
    )
