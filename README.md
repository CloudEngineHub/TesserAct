<p align="center">
  <h1 align="center">TesserAct: Learning 4D Embodied World Models</h1>
  <p align="center">
    ICCV 2025
  </p>
  <p align="center">
    <a href="https://haoyuzhen.com">Haoyu Zhen*</a>,
    <a href="https://qiaosun22.github.io/">Qiao Sun*</a>,
    <a href="https://icefoxzhx.github.io/">Hongxin Zhang</a>,
    <a href="https://senfu.github.io/">Junyan Li</a>,
    <a href="https://rainbow979.github.io/">Siyuan Zhou</a>,
    <a href="https://yilundu.github.io/">Yilun Du</a>,
    <a href="https://people.csail.mit.edu/ganchuang">Chuang Gan</a>
  </p>
  <p align="center">
    <a href="https://arxiv.org/abs/2504.20995">
      <img src='https://img.shields.io/badge/Paper-PDF-red?style=flat&logo=arXiv&logoColor=red' alt='Paper PDF'>
    </a>
    <a href='https://tesseractworld.github.io' style='padding-left: 0.5rem;'>
      <img src='https://img.shields.io/badge/Project-Page-blue?style=flat&logo=Google%20chrome&logoColor=blue' alt='Project Page'>
    </a>
    <a href='https://huggingface.co/anyeZHY/tesseract' style='padding-left: 0.5rem;'>
      <img src='https://img.shields.io/badge/Model-Hugging%20Face-yellow?style=flat&logo=Hugging%20face&logoColor=yellow' alt='Model Hugging Face'>
    </a>
  </p>
</p>

We propose TesserAct, **the first open-source and generalized 4D World Model for robotics**, which takes input images and text instructions to generate RGB, depth, and normal videos, reconstructing a 4D scene and predicting actions.

<p align="center">
    <img src="asset/teaser.png" alt="Logo" width="190%">
</p>

<br>

<!-- TABLE OF CONTENTS -->
<details open="open" style='padding: 10px; border-radius:5px 30px 30px 5px; border-style: solid; border-width: 1px;'>
  <summary>Tabel of Contents</summary>
  <ol>
    <li>
      <a href="#installation">Installation</a>
    </li>
    <li>
      <a href="#data-preparation">Data Preparation</a>
    </li>
    <li>
      <a href="#training">Training</a>
        <ul>
          <li>
            <a href="#pre-training-or-full-fine-tuning">Pre-training or Full Fine-tuning</a>
          </li>
          <li>
            <a href="#lora-fine-tuning">LoRA Fine-tuning</a>
          </li>
        </ul>
    </li>
    <li>
      <a href="#inference">Inference</a>
    </li>
    <li>
      <a href="#citation">Citation</a>
    </li>
    <li>
      <a href="#acknowledgement">Acknowledgement</a>
    </li>
  </ol>
</details>

## News
- [2025-06-25] TesserAct is accepted to ICCV 2025!
- [2025-06-19] We provide an efficient RGB+Depth+Normal LoRA fine-tuning script for custom datasets.
- [2025-06-18] We provide a RGB-only LoRA inference script that achieves the best generalization ability for robotics video generation.
- [2025-06-06] We have released the training code and data generation scripts!
- [2025-05-05] We have updated the gallery and added more results on the [project website](https://tesseractworld.github.io).
- [2025-05-04] We add [USAGE.MD](doc/usage.md) to provide more details about the models and how to use the models on your own data!
- [2025-04-29] We have released the inference code and TesserAct-v0.1 model weights!

## Installation
Create a conda environment and install the required packages:
```bash
conda create -n tesseract python=3.9
conda activate tesseract
pip install -r requirements.txt

git clone https://github.com/UMass-Embodied-AGI/TesserAct.git
cd TesserAct
pip install -e .
```

## Data Preparation
Please refer to [DATA.md](DATA.md) for data generation scripts and dataset preparation.

## Training
### Pre-training or Full Fine-tuning
To pre-train the full TesserAct model from CogVideoX, we provide a training script based on [Finetrainers](https://github.com/a-r-r-o-w/finetrainers). The training code supports distributed training with multiple GPUs or multi-nodes.

To pre-train our TesserAct model, run the following command:
```bash
bash train_i2v_depth_normal_sft.sh
```

To fine-tune our released TesserAct model, modify the model loading code in [tesseract/i2v_depth_normal_sft.py](tesseract/i2v_depth_normal_sft.py):
```python
transformer = CogVideoXTransformer3DModel.from_pretrained_modify(
    "anyeZHY/tesseract",
    subfolder="tesseract_v01e_rgbdn_sft",
    ...
)
```

### LoRA Fine-tuning

You can efficiently fine-tune our TesserAct model using LoRA (Low-Rank Adaptation) with your own data (~100 videos). This approach requires approximately **~30GB GPU memory** and allows for efficient training (~2 days) on custom datasets.

To fine-tune using LoRA, run the following command:
```bash
bash train_i2v_depth_normal_lora.sh
```

> [!WARNING]
> LoRA fine-tuning is experimental and not fully tested yet.

> [!NOTE]
> We will give a detailed training guide in the future: why TesserAct has better generalization, how to set the hyperparameters and performance between different training methods (SFT vs LoRA).
>
> We don't have a clear plan for releasing the whole dataset yet, because depth data is usually stored as floats, which takes up a lot of space and makes uploading to Hugging Face very difficult. However, we've provided scripts to show how to prepare the data.

## Inference

Now TesserAct includes following models. The names of the models are in the format of `anyeZHY/tesseract/` (huggingface repo name) + `<model_name>_<version>_<modality>_<training_method>`. In `<version>`, postfix `p` indicates the model is production-ready and `e` means the model is experimental. We will keep updating the model weights and scaling the dataset to improve the performance of the models.
```
anyeZHY/tesseract/tesseract_v01e_rgbdn_sft
anyeZHY/tesseract/tesseract_v01e_rgb_lora
```

> [!IMPORTANT] 
> It is recommended to read [USAGE.MD](doc/usage.md) for more details **before running the inference code on your own data.**
We provide a guide on how to prepare inputs, such as text prompt. We also analyze the model's limitations and performance, including:
>
> - Tasks that the model can reliably accomplish.
>
> - Tasks that are achievable but with certain success rates. In the future, this may be improved by using techniques like test-time scaling.
>
> - Tasks that are currently beyond the model's capabilities.

You can run the inference code with the following command (Optional flags: `--memory_efficient`).
```bash
python inference/inference_rgbdn_sft.py \
  --weights_path anyeZHY/tesseract/tesseract_v01e_rgbdn_sft \
  --image_path asset/images/fruit_vangogh.png \
  --prompt "pick up the apple google robot"
```
This inference code will generate a video of the google robot picking up the apple in the Van Gogh Painting. Try other prompts like `pick up the pear Franka Emika Panda`! Or `asset/images/majo.jpg` with prompt `Move the cup near bottle Franka Emika Panda`!

For RGB-only generation using the LoRA model, you can use:
```bash
python inference/inference_rgb_lora.py \
  --weights_path anyeZHY/tesseract/tesseract_v01e_rgb_lora \
  --image_path asset/images/fruit_vangogh.png \
  --prompt "pick up the apple google robot"
```
The RGB LoRA model offers the best generalization quality for RGB video generation, making it ideal for diverse robotic manipulation tasks.

For RGB+Depth+Normal generation using the LoRA model, you can use:
```bash
python inference/inference_rgbdn_lora.py \
  --base_weights_path anyeZHY/tesseract/tesseract_v01e_rgbdn_sft \
  --lora_weights_path ./your_local_lora_weights \
  --image_path asset/images/fruit_vangogh.png \
  --prompt "pick up the apple google robot"
```

You may find output videos in the `results` folder.
Note: When we test the model on another server, the results are exactly the same as those we uploaded to GitHub.
So if you find they are different and get unexpected results like noisy videos, please check your environment and the version of the packages you are using.

> [!WARNING]
> Because RT1 and Bridge normal data is generated by [Temporal Marigold](https://huggingface.co/docs/diffusers/en/using-diffusers/marigold_usage#frame-by-frame-video-processing-with-temporal-consistency), sometimes normal outputs are not perfect. We are working on improving the data using [NormalCrafter](https://github.com/Binyr/NormalCrafter).

Below is a list of TODOs for the inference part.
- [x] LoRA inference code
- [ ] Blender rendering code (check package [PyBlend](https://github.com/anyeZHY/PyBlend)!)
- [ ] Normal Integration

## Citation
If you find our work useful, please consider citing:
```bibtex
@article{zhen2025tesseract,
  title={TesserAct: Learning 4D Embodied World Models}, 
  author={Haoyu Zhen and Qiao Sun and Hongxin Zhang and Junyan Li and Siyuan Zhou and Yilun Du and Chuang Gan},
  year={2025},
  eprint={2504.20995},
  archivePrefix={arXiv},
  primaryClass={cs.CV},
  url={https://arxiv.org/abs/2504.20995}, 
}
```

## Acknowledgements
We would like to thank the following works for their code and models:
- Training: [CogVideo](https://github.com/THUDM/CogVideo), [Finetrainers](https://github.com/a-r-r-o-w/finetrainers) and [VideoX-Fun](https://github.com/aigc-apps/VideoX-Fun)
- Data generation: [RollingDepth](https://github.com/prs-eth/rollingdepth), [Marigold](https://github.com/prs-eth/Marigold) and [DSINE](https://github.com/baegwangbin/DSINE)
- Datasets: [OpenX](https://robotics-transformer-x.github.io/), [RLBench](https://github.com/stepjam/RLBench), [Hiveformer](https://github.com/vlc-robot/hiveformer) and [Colosseum](https://github.com/robot-colosseum/robot-colosseum)
- Why normals: [BiNI](https://github.com/xucao-42/NormalIntegration), [ICON](https://github.com/YuliangXiu/ICON), [StableNormal](https://github.com/Stable-X/StableNormal) and [NormalCrafter](https://github.com/Binyr/NormalCrafter)

We are extremely grateful to Pengxiao Han for assistance with the baseline code, and to Yuncong Yang, Sunli Chen,
Jiaben Chen, Zeyuan Yang, Zixin Wang, Lixing Fang, and many other friends in our [Embodied AGI Lab](https://embodied-agi.cs.umass.edu/)
for their helpful feedback and insightful discussions.
