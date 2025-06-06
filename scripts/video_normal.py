import os
import shutil
from tqdm import tqdm
import torch
import random
import diffusers
import numpy as np
from PIL import Image
import imageio.v3 as imageio
import argparse
import torch.multiprocessing as mp
from functools import partial


def process_video(scene_id, data_path, pipe, latent_common, device):
    """Process a single video using the provided model."""
    scene_path = os.path.join(data_path, scene_id)
    path_in = os.path.join(scene_path, "video", "rgb.mp4")
    path_out = os.path.join(scene_path, "video", "normal.mp4")

    # Check if output file exists and can be opened
    if os.path.exists(path_out):
        try:
            # Try to open the file to verify it's valid
            with open(path_out, "rb") as f:
                # If we can open the file, skip processing
                return
        except Exception:
            # If file exists but can't be opened, remove it and continue processing
            os.remove(path_out)

    # Read frames from MP4 file
    reader = imageio.imiter(path_in)
    frames = [frame for frame in reader]

    last_frame_latent = None
    first_frame_latent = None

    out = []
    for frame_id, frame in enumerate(frames):
        frame = Image.fromarray(frame)

        latents = latent_common
        if last_frame_latent is not None:
            latents = 0.9 * latents + 0.1 * last_frame_latent

        depth = pipe(
            frame,
            match_input_resolution=True,
            latents=latents,
            output_latent=True,
            ensemble_size=1,
        )

        if first_frame_latent is None:
            first_frame_latent = depth.latent

        last_frame_latent = depth.latent

        out.append(pipe.image_processor.visualize_normals(depth.prediction)[0])

    # Save video using imageio.v3
    imageio.imwrite(path_out, out, fps=30)


def process_videos_on_gpu(scene_list, data_path, device_id, args):
    """Process multiple videos on a specific GPU."""
    # Set device for this process
    device = torch.device(f"cuda:{device_id}")
    torch.cuda.set_device(device)

    # Initialize model on this device
    print(f"Loading pipeline components on GPU {device_id}...")
    pipe = diffusers.MarigoldNormalsPipeline.from_pretrained(
        "prs-eth/marigold-normals-lcm-v0-1", variant="fp16", torch_dtype=torch.float16
    ).to(device=device)
    pipe.set_progress_bar_config(disable=True)

    # Initialize latent common
    if "bridge" in data_path:
        size = (640, 480)
        latent_common = torch.randn((1, 4, 768 * size[1] // (8 * max(size)), 768 * size[0] // (8 * max(size)))).to(
            device=device, dtype=torch.float16
        )
    elif "fractal" in data_path:
        size = (320, 256)
        latent_common = torch.randn((1, 4, 768 * size[1] // (8 * max(size)) + 1, 768 * size[0] // (8 * max(size)))).to(
            device=device, dtype=torch.float16
        )
    else:
        raise ValueError("Unknown data path")

    # Process all assigned videos using the same model
    for scene_id in tqdm(scene_list, desc=f"Processing videos on GPU {device_id}"):
        process_video(scene_id, data_path, pipe, latent_common, device)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="bridge")
    parser.add_argument(
        "--num_gpus", type=int, default=None, help="Number of GPUs to use. If None, uses all available GPUs."
    )
    args = parser.parse_args()

    # seed everything
    seed = 23
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    data_path = f"data/{args.dataset}/processed"
    scene_list = sorted(os.listdir(data_path), key=lambda x: int(x))

    # Determine number of GPUs to use
    if not torch.cuda.is_available():
        print("CUDA is not available. Running on CPU will be slow.")
        num_gpus = 0
    else:
        num_gpus = args.num_gpus if args.num_gpus is not None else torch.cuda.device_count()
        print(f"Using {num_gpus} GPUs")

    if num_gpus > 1:
        # Create process pool
        mp.set_start_method("spawn", force=True)
        with mp.Pool(num_gpus) as pool:
            # Distribute scenes evenly across GPUs
            scenes_per_gpu = len(scene_list) // num_gpus
            remaining_scenes = len(scene_list) % num_gpus

            start_idx = 0
            results = []
            for gpu_id in range(num_gpus):
                # Calculate number of scenes for this GPU
                num_scenes = scenes_per_gpu + (1 if gpu_id < remaining_scenes else 0)
                gpu_scenes = scene_list[start_idx : start_idx + num_scenes]
                start_idx += num_scenes

                results.append(pool.apply_async(process_videos_on_gpu, args=(gpu_scenes, data_path, gpu_id, args)))

            # Wait for all processes to complete
            for result in results:
                result.get()
    else:
        # Single GPU or CPU processing
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        process_videos_on_gpu(scene_list, data_path, 0, args)


if __name__ == "__main__":
    main()
