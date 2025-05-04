import os
import diffusers
import torch
import argparse
import numpy as np

parser = argparse.ArgumentParser(description="Inference script for Marigold Depth and Normals")
parser.add_argument(
    "--image_folder", type=str, required=True, help="Path to the input image for depth and normals estimation"
)
args = parser.parse_args()

image_list = os.listdir(args.image_folder)
image_list = [
    img for img in image_list if img.endswith((".png", ".jpg")) and not "normal" in img and not "depth" in img
]

normal_pipe = diffusers.MarigoldNormalsPipeline.from_pretrained(
    "prs-eth/marigold-normals-v1-1", variant="fp16", torch_dtype=torch.float16
).to("cuda")
depth_pipe = diffusers.MarigoldDepthPipeline.from_pretrained(
    "prs-eth/marigold-depth-v1-1", variant="fp16", torch_dtype=torch.float16
).to("cuda")

for image_name in image_list:
    # ======= Load image ========
    image_path = os.path.join(args.image_folder, image_name)
    image_name_ = image_path.replace(".png", "").replace(".jpg", "")
    normal_to_save = image_name_ + "_normal.png"
    depth_to_save = image_name_ + "_depth.npy"
    if os.path.exists(normal_to_save) and os.path.exists(depth_to_save):
        print(f"Skipping {image_name} as normals and depth already exist.")
        continue
    image = diffusers.utils.load_image(image_path)

    # ======== Normals ========
    normals = normal_pipe(image)
    vis = normal_pipe.image_processor.visualize_normals(normals.prediction)
    vis[0].save(normal_to_save)

    # ======== Depth ========
    depth = depth_pipe(image, output_type="np")
    depth_npy = depth.prediction
    depth_npy = (depth_npy - depth_npy.min()) / (depth_npy.max() - depth_npy.min())
    np.save(depth_to_save, depth_npy[0])
