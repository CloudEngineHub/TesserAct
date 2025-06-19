import os
import gc
import torch
import cv2
import numpy as np
from diffusers.utils import load_image, export_to_video
from diffusers import CogVideoXDPMScheduler

# export PYTHONPATH=$PYTHONPATH:./
from tesseract.modules.tesseract_pipeline import TesserActImageToDepthNormalVideoPipeline
from tesseract.modules.tesseract_model import TesserActDepthNormal
from tesseract.utils import print_memory, crop_and_resize_frames
from huggingface_hub import snapshot_download

torch.set_grad_enabled(False)
# seed everything
seed = 23
torch.manual_seed(seed)
np.random.seed(seed)


# --------------------------------------
# Inference function
# --------------------------------------
def validate(
    pretrained_model_path: str,
    base_weights_path: str,
    lora_weights_path: str,
    validation_prompts: list,
    validation_images: list,
    output_dir: str,
    num_inference_steps: int = 50,
    guidance_scale: float = 6.0,
    image_guidance_scale: float = 6.0,
    use_dynamic_cfg: bool = True,
    height: int = 256,
    width: int = 320,
    num_validation_videos: int = 1,
    fps: int = 8,
    mixed_precision: str = "bf16",
    memory_efficient: bool = False,
):
    # --------------------------------------
    # 1. Set up precision and device
    # --------------------------------------
    if mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif mixed_precision == "bf16":
        weight_dtype = torch.bfloat16
    else:
        weight_dtype = torch.float32
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # --------------------------------------
    # 2. Load the pipeline
    # --------------------------------------
    print("Loading pipeline...")
    pipe = TesserActImageToDepthNormalVideoPipeline.from_pretrained(pretrained_model_path, torch_dtype=weight_dtype).to(
        device
    )
    
    # --------------------------------------
    # 3. Load the base RGBD+N model
    # --------------------------------------
    print("Loading base RGBD+N transformer checkpoint...")
    if os.path.exists(base_weights_path):
        subfolder = None
    else:
        subfolder = base_weights_path.split("/")[-1]
        base_weights_path = "/".join(base_weights_path.split("/")[:-1])
    print(f"Loading base weights from {base_weights_path}, subfolder: {subfolder}")
    transformer = (
        TesserActDepthNormal.from_pretrained_modify(
            base_weights_path,
            subfolder=subfolder,
        )
        .to(dtype=weight_dtype, device=device)
        .eval()
    )
    transformer.config.in_channels += 16 * 4
    del transformer.patch_embed.pos_embedding
    transformer.patch_embed.use_learned_positional_embeddings = False
    transformer.config.use_learned_positional_embeddings = False
    pipe.transformer = transformer

    # --------------------------------------
    # 4. Load LoRA weights
    # --------------------------------------
    print("Loading LoRA weights...")
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
    pipe.set_adapters(["cogvideox-lora"], [1.0])

    # Load DPMScheduler or your custom scheduler config
    pipe.scheduler = CogVideoXDPMScheduler.from_config(pipe.scheduler.config)

    if memory_efficient:
        pipe.vae.enable_slicing()
        pipe.vae.enable_tiling()
        pipe.enable_model_cpu_offload()

    for im_idx, (prompt, image_path) in enumerate(zip(validation_prompts, validation_images)):
        print(f"Running validation for prompt: {prompt}")
        print(f"Using image: {image_path}")
        validation_image = image_path
        if validation_image.endswith(".mp4"):
            cap = cv2.VideoCapture(validation_image)
            ret, val_image = cap.read()
            cap.release()
            val_image = cv2.cvtColor(val_image, cv2.COLOR_BGR2RGB)
        else:
            val_image = load_image(image_path)  # PIL image
        val_image = crop_and_resize_frames([np.array(val_image).astype(np.float32)], (height, width))[0]
        val_image = torch.from_numpy(val_image).to(dtype=weight_dtype, device=device)
        val_image = val_image / 255.0  # [0, 1]

        # ==== load depth image ====
        depth_path = validation_image.replace(".png", "").replace(".jpg", "") + "_depth.npy"
        depth_image = np.load(depth_path)
        depth_image = 1 - depth_image
        depth_image = crop_and_resize_frames([depth_image], (height, width))[0]
        depth_image = torch.from_numpy(depth_image[..., None]).to(dtype=weight_dtype, device=device)  # [H, W, 1]
        depth_image = depth_image.repeat(1, 1, 3)  # [H, W, 3]
        if depth_image.min() < 0:
            depth_image = (depth_image + 1.0) / 2.0  # [0, 1]

        # ==== load normal image ====
        normal_image = validation_image.replace(".png", "").replace(".jpg", "") + "_normal.png"
        normal_image = cv2.cvtColor(cv2.imread(normal_image), cv2.COLOR_BGR2RGB)

        normal_image = crop_and_resize_frames([normal_image], (height, width))[0]
        normal_image = torch.from_numpy(normal_image).to(dtype=weight_dtype, device=device)
        normal_image = normal_image / 255.0  # [0, 1]
        image = torch.cat([val_image, depth_image, normal_image], dim=2)  # [H, W, 9]
        image = image.permute(2, 0, 1).unsqueeze(0)  # [B, C, H, W]

        for idx in range(num_validation_videos):
            gc.collect()
            torch.cuda.empty_cache()
            result = pipe(
                image=image,
                prompt=prompt,
                guidance_scale=guidance_scale,
                image_guidance_scale=image_guidance_scale,
                use_dynamic_cfg=use_dynamic_cfg,
                height=height,
                width=width,
                num_inference_steps=num_inference_steps,
                num_frames=49,
            )
            print_memory(device)
            video_frames = result.frames[0]

            # Save the video
            safe_prompt = prompt.replace(" ", "_")
            output_file = os.path.join(output_dir, f"val_{im_idx}_{safe_prompt}_{idx}.mp4")
            export_to_video(video_frames, output_file, fps=fps)
            print(f"Saved validation video to: {output_file}")


# Example usage
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--base_weights_path", type=str, default="anyezhy/tesseract/tesseract_v01e_rgbdn_sft")
    parser.add_argument("--lora_weights_path", type=str, default="./local_lora_weights")
    parser.add_argument("--image_path", type=str, default="asset/images/fruit_vangogh.png")
    parser.add_argument("--prompt", type=str, default="pick up the apple google robot")
    parser.add_argument("--memory_efficient", action="store_true", default=False)
    args = parser.parse_args()

    pretrained_model = "THUDM/CogVideoX-5b-I2V"  # always use this model
    base_weights_path = args.base_weights_path
    lora_weights_path = args.lora_weights_path

    val_images = [args.image_path]
    val_prompts = [args.prompt]

    out_dir = "./results"
    os.makedirs(out_dir, exist_ok=True)

    with torch.no_grad():
        validate(
            pretrained_model_path=pretrained_model,
            base_weights_path=base_weights_path,
            lora_weights_path=lora_weights_path,
            validation_prompts=val_prompts,
            validation_images=val_images,
            output_dir=out_dir,
            num_inference_steps=50,
            guidance_scale=7.5,
            image_guidance_scale=1.5,
            use_dynamic_cfg=False,
            height=480,
            width=640,
            num_validation_videos=2,
            fps=8,
            mixed_precision="bf16",
            memory_efficient=args.memory_efficient,
        )
