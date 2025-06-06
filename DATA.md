# TesserAct Data Processing Pipeline

This document describes the setup and execution of the TesserAct data processing pipeline for bridge dataset processing.

## Environment Setup

First, create and activate a new conda environment with Python 3.10. This environment will be used for all data processing tasks.

```bash
conda create -n tesseract-data python=3.10
conda activate tesseract-data
```

Next, install the required packages. This includes the base requirements and the RollingDepth package which is needed for depth estimation.

```bash
pip install -r requirements.txt
pip install git+https://github.com/anyeZHY/rollingdepth.git
```

Finally, install the modified diffusers package from prs-eth. This package contains custom modifications required for the RollingDepth implementation.

```bash
# Install diffusers modified by prs-eth
# Follow instructions at: https://github.com/prs-eth/RollingDepth/blob/main/script/install_diffusers_dev.sh
```

## Data Processing Pipeline

The pipeline consists of three main steps:

### Download Raw Bridge Dataset

Download the raw bridge dataset. This dataset contains the original demonstrations that will be processed.

```bash
wget https://rail.eecs.berkeley.edu/datasets/bridge_release/data/demos_8_17.zip
mv demos_8_17.zip data
cd data && unzip demos_8_17.zip && cd ..
```

### Bridge Dataset Preprocessing

The preprocessing script processes the raw bridge dataset by finding language files with specific keywords, converting image sequences to videos, and organizing the data in a structured format. The processed data will be saved in the data/bridge/processed directory.

```bash
python scripts/preprocess_bridge.py
```

### Video Depth Estimation

The depth estimation script processes the videos to generate depth maps using the RollingDepth model. It supports multiple GPUs for parallel processing and saves the depth predictions as compressed NPZ files. The output will be saved in the `depth/npz` directory for each scene.

```bash
python scripts/video_depth.py -i data/bridge/processed -o data/bridge/processed --verbose
```

### Video Normal Map Generation

The normal map generation script creates normal maps for each video using the MarigoldNormals model. It supports multi-GPU processing and saves the output as normal.mp4 in each scene's video directory. You can specify the number of GPUs to use with the --num_gpus parameter.

```bash
python scripts/video_normal.py --dataset bridge --num_gpus 8
```

In the future, we will use the NormalCrafter model to generate normal maps.

## Directory Structure

After processing, the data will be organized in the following structure. Each scene will have its own directory containing the processed video, depth maps, and instruction file.

```
data/bridge/processed/
├── <scene_id>/
│   ├── video/
│   │   ├── rgb.mp4
│   │   └── normal.mp4
│   ├── depth/
│   │   └── npz/
│   │       └── depth.npz
│   └── instruction.txt
```
