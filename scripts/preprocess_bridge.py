#!/usr/bin/env python3
"""
Bridge dataset preprocessing script.

This script processes the bridge dataset by:
1. Finding all language files containing specific keywords
2. Converting image sequences to videos
3. Organizing the processed data in a structured format

Usage:
    python scripts/preprocess_bridge.py
"""

import os
import logging
import imageio.v2 as imageio
import glob
import json
from tqdm import tqdm

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


class BridgePreprocessor:
    def __init__(
        self,
        raw_data_dir="data/raw",
        output_dir="data/bridge/processed",
        cache_dir="cache",
        required_keywords=[],
        excluded_phrases=[],
        fps=30,
    ):
        self.raw_data_dir = raw_data_dir
        self.output_dir = output_dir
        self.cache_dir = cache_dir
        self.required_keywords = required_keywords
        self.excluded_phrases = excluded_phrases
        self.fps = fps

        # Create necessary directories
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(self.cache_dir, exist_ok=True)

    def _get_lang_files(self):
        cache_file = os.path.join(self.cache_dir, "bridge_raw_lang_files.json")

        if not os.path.exists(cache_file):
            logger.info("Finding all lang.txt files...")
            lang_files = glob.glob(os.path.join(self.raw_data_dir, "**/lang.txt"), recursive=True)
            with open(cache_file, "w") as f:
                json.dump(lang_files, f)
        else:
            with open(cache_file, "r") as f:
                lang_files = json.load(f)

        return lang_files

    def _process_instruction(self, content):
        content = content.lower().strip().split("\n")[0]

        # Check required keywords
        if not all(keyword in content for keyword in self.required_keywords):
            return None

        # Check excluded phrases
        if any(phrase in content for phrase in self.excluded_phrases):
            return None

        return content

    def _process_image_sequence(self, img_dir, scene_dir, instruction):
        # Create scene directories
        video_dir = os.path.join(scene_dir, "video")
        os.makedirs(video_dir, exist_ok=True)

        # Save instruction
        with open(os.path.join(scene_dir, "instruction.txt"), "w") as f:
            f.write(instruction)

        # Get and sort images
        images = sorted(
            glob.glob(os.path.join(img_dir, "im_*.jpg")),
            key=lambda x: int(os.path.splitext(os.path.basename(x))[0].split("_")[-1]),
        )

        if not images:
            logger.warning(f"No images found in {img_dir}")
            return False

        # Process images
        frames = [imageio.imread(img) for img in images]
        output_video = os.path.join(video_dir, "rgb.mp4")
        imageio.mimsave(output_video, frames, fps=self.fps)

        return True

    def process(self):
        lang_files = self._get_lang_files()
        scene_counter = 0

        for lang_file in tqdm(lang_files, desc="Processing scenes"):
            # Read and process instruction
            with open(lang_file, "r") as f:
                content = f.read()

            instruction = self._process_instruction(content)
            if not instruction:
                continue

            # Process image directories
            base_dir = os.path.dirname(lang_file)
            image_dirs = sorted(glob.glob(os.path.join(base_dir, "images*")))

            for img_dir in image_dirs:
                scene_dir = os.path.join(self.output_dir, str(scene_counter))
                if self._process_image_sequence(img_dir, scene_dir, instruction):
                    scene_counter += 1

        logger.info(f"Processing complete. Processed {scene_counter} scenes.")


def main():
    preprocessor = BridgePreprocessor()
    preprocessor.process()


if __name__ == "__main__":
    main()
