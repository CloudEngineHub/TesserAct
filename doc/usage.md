# Usage

## Model Cards
We provide two model cards for TesserAct, which are available on Hugging Face.

| Model Name | Description | Memory Usage | Training Data |
|------------|-------------|--------------|---------------|
| tesseract_v01e_rgbdn_sft | RGB Depth Normal video generation, Finetuned fro CogVideoX-5B, trained for 40000 steps | 25-29G | RT1, Bridge, RLBench |
| tesseract_v01e_rgb_lora | RGB video generation with LoRA, trained for 12000 steps | 25-29G | RT1, Bridge, RLBench |

## Prepare the Input
### A. Design your prompt
TesserAct was trained on inputs in the format: [Instruction] + [Robot Name]. For now, we only support the following robot names: `google robot`, `Franka Emika Panda` and `Trossen WidowX 250 robot arm`. So please make sure to include the robot name in your prompt. For example: `pick up the apple google robot` or `move the mug near the bottle Franka Emika Panda`.

For better results, you can use an instruction prompt that is closer to those in the training dataset. But this is not necessary. The model is quite robust and can handle a variety of instructions. For example, you can use `Pick Pickle Rick Franka Emika Panda`.

### B. Generate Depth and Normal Annotations
TesserAct takes the RGB-DN image and text instruction as input.
So for any in-the-wild image from the internet, you need to first estimate depth and normal map by Marigold. Below are the steps to do so and call the model.
1. Put all the images in a folder (e.g. `data/images`). Please avoid using images in which the manipulated objects are too large.
2. Run Marigold to generate depth and normal maps:
   ```bash
   python inference/inference_marigold.py --image_folder data/images
   ```
   This will generate `*_normal.png` and `*_depth.npy` files in the same folder.
3. Run TesserAct to generate the video. We will resize the image to 640x480 automatically. Other resolutions may not work well now. In the future, we will release a new version of the model that can handle any resolution.
   ```bash
   python inference/inference_rgbdn_sft.py \
       --weights_path anyeZHY/tesseract/tesseract_v01e_rgbdn_sft \
       --image_path rgb_image.png \
       --prompt "your prompt here"
   ```
    This will generate a video in the `results` folder.

If you have a depth and normal map already from other sources instead of Marigold, you can skip step 2 and directly run TesserAct with the depth and normal map files. But make sure:
- The range of the depth map is between 0 and 1. The farther the object is, the smaller the value. Stored in a 1-channel float Numpy array.
- The normal map is stored in a 3-channel PNG format. The normal vector is stored in the RGB channels, where R, G, and B represent the X, Y, and Z components of the normal vector respectively. For more details you can check [marigold_usage#normals](https://huggingface.co/docs/diffusers/en/using-diffusers/marigold_usage#surface-normals-estimation) and [visualize_normals](https://huggingface.co/docs/diffusers/v0.33.1/en/api/pipelines/marigold#diffusers.pipelines.marigold.MarigoldImageProcessor.visualize_normals). (E.g., if you are using [Depth-to-Normal](https://github.com/baegwangbin/DSINE/tree/main/utils/d2n) in DSINE, you need to flip the value of the X-axis to get the correct normal map.)

## Results
See our [gallery](https://tesseractworld.github.io/) for examples of the results produced by the models.

## Limitations and Failure Cases

> [!WARNING]
> This section is still under construction.