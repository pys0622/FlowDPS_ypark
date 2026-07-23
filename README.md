# [ICCV2025] FlowDPS: Flow-Driven Posterior Sampling for Inverse Problems

![img](assets/main.jpg)

## Abstract


❗️Flow matching is a recent state-of-the-art framework for generative modeling based on ordinary differential equations (ODEs). While closely related to diffusion models, __it provides a more general perspective__ on generative modeling. 

❓ Although inverse problem solving has been extensively explored using diffusion models, it has not been rigorously examined within the broader context of flow models. Therefore, __we extend diffusion inverse solvers (DIS)— which perform posterior sampling by combining a denoising diffusion prior with a likelihood gradient—into the flow framework.__

👍 Our proposed framework, Flow-Driven Posterior Sampling (FlowDPS), can also be seamlessly integrated into a latent flow model with a transformer architecture. Across four linear inverse problems, we confirm that FlowDPS outperforms state-of-the-art alternatives, all without requiring additional training.


## Quick Start

### Environment Setup

First, clone this repository and install requirements.

```
git clone https://github.com/FlowDPS-Inverse/FlowDPS.git
cd FlowDPS
conda create -n flowdps python==3.10
conda activate flowdps
pip install -r requirements.txt
```

> The provided requirements.txt installs torch with CUDA 11.8. If you are using other versions, please change it.

For the motion blur problem, clone the repository below.
```
git clone https://github.com/LeviBorodenko/motionblur.git
```

### +) Download Datasets as Below
```
wget -c http://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_train_HR.zip ;
unzip -q DIV2K_train_HR.zip
```
```
wget -N https://www.dropbox.com/s/t9l9o3vsx2jai3z/afhq.zip?dl=0 -O afhq.zip ;
unzip afhq.zip ;
rm afhq.zip
```
```
git clone https://github.com/NVlabs/ffhq-dataset.git ;
unfunction gdown ;
python -m pip install -U gdown ;
python -m gdown \
  16N0RV4fHI6joBuKbQAoG34V_cQk7vxSA \
  -O ffhq-dataset-v2.json ; 
python ffhq-dataset/download_ffhq.py -i;
```
### +) Sample & Resize Images for Evaluation as Below
```
python utils/prepare_dataset.py \
  --input datasets/raw/AFHQ/afhq/val \
  --output datasets/prepared/AFHQ_val1000 \
  --num_samples 1000 \
  --seed 0 \
  --selection first \
  --resize_mode flowdps \
  --afhq_catndog_only \
  --img_size 768
```
```
python utils/prepare_dataset.py \
  --input datasets/raw/DIV2K/DIV2K_train_HR \
  --output datasets/prepared/DIV2K_train800 \
  --num_samples 800 \
  --seed 0 \
  --selection first \
  --resize_mode flowdps \
  --img_size 768
```
```
python utils/prepare_dataset.py \
  --input datasets/raw/FFHQ/images1024x1024 \
  --output datasets/prepared/FFHQ_val1000 \
  --num_samples 1000 \
  --seed 0 \
  --selection random \
  --ffhq_validation_only \
  --resize_mode flowdps \
  --img_size 768
```
### +) Save Measurement as .pt
```
python -m utils.prepare_measurement \
    --img_path datasets/prepared/{AFHQ_val1000, DIV2K_train800, FFHQ_val1000}/images \
    --task sr_avgpool \
    --deg_scale 12 \
    --noise_std 0.03 \
    --seed 0
```
```
python -m utils.prepare_measurement \
    --img_path datasets/prepared/{AFHQ_val1000, DIV2K_train800, FFHQ_val1000}/images \
    --task sr_bicubic \
    --deg_scale 12 \
    --noise_std 0.03 \
    --seed 0
```
```
python -m utils.prepare_measurement \
    --img_path datasets/prepared/{AFHQ_val1000, DIV2K_train800, FFHQ_val1000}/images \
    --task deblur_motion \
    --deg_scale 61 \
    --noise_std 0.03 \
    --seed 0
```
Add --save_preview for DIV2K, in order to save input image for prompt generation

### +) Extract Prompt Based on Image Manifest File / Measurement
#### Installing SeeSR for DAPE prompt extraction
```
cd .. ;
git clone https://github.com/cswry/SeeSR.git ;
cd SeeSR ;
pip install pytorch_lightning \
    loralib \
    fairscale \
    pydantic==1.10.11 \
    gradio==3.24.0 \
    timm ;
mkdir -p preset/models ;
wget -O preset/models/ram_swin_large_14m.pth \
    https://huggingface.co/xinyu1205/recognize_anything_model/resolve/main/ram_swin_large_14m.pth ;
python -m gdown 127swnIotVkbl2nDnrXBpKtx4_AnFMhtR \
    -O preset/models/DAPE.pth
```
####    CLI for extraction

```
python utils/prepare_prompt.py \
    --dataset AFHQ \
    --manifest datasets/prepared/AFHQ_val1000/manifest.json \
    --output datasets/prepared/AFHQ_val1000/prompts.txt
```
```
python utils/prepare_prompt.py \
    --dataset FFHQ \
    --manifest datasets/prepared/FFHQ_val1000/manifest.json \
    --output datasets/prepared/FFHQ_val1000/prompts.txt
```
```
python utils/prepare_prompt.py \
    --dataset DIV2K \
    --manifest datasets/prepared/DIV2K_train800/manifest.json \
    --measurement_preview_dir datasets/prepared/DIV2K_train800/measurement_preview/deblur_motion \
    --seesr_root ../SeeSR \
    --output datasets/prepared/DIV2K_train800/prompts_deblur_motion.txt \
		| tee datasets/prepared/DIV2K_train800/prepare_prompt.deblur_motion.log ;
python utils/prepare_prompt.py \
    --dataset DIV2K \
    --manifest datasets/prepared/DIV2K_train800/manifest.json \
    --measurement_preview_dir datasets/prepared/DIV2K_train800/measurement_preview/sr_avgpool_x12 \
    --seesr_root ../SeeSR \
    --output datasets/prepared/DIV2K_train800/prompts_sr_avgpool_x12.txt \
		| tee datasets/prepared/DIV2K_train800/prepare_prompt.sr_avgpool_x12.log ;
		
python utils/prepare_prompt.py \
    --dataset DIV2K \
    --manifest datasets/prepared/DIV2K_train800/manifest.json \
    --measurement_preview_dir datasets/prepared/DIV2K_train800/measurement_preview/sr_bicubic_x12 \
    --seesr_root ../SeeSR \
    --output datasets/prepared/DIV2K_train800/prompts_sr_bicubic_x12.txt \
		| tee datasets/prepared/DIV2K_train800/prepare_prompt.sr_bicubic_x12.log ;
```

### +) Inference for Precalculated Measurements
```
python solve.py \
    --workdir experiments/0721_afhq_sr_avgpool \
    --measurement_path datasets/prepared/AFHQ_val1000/measurement/sr_avgpool_x12 \
    --num_samples 10 \
    --prompt_file datasets/prepared/AFHQ_val1000/prompts.txt \
    --task sr_avgpool \
    --deg_scale 12 \
    --efficient_memory;
```
### +) Evaluation
```
python eval.py \
    --path1 datasets/prepared/AFHQ_val1000/images \
    --path2 experiments/0721_afhq_deblur_motion/recon \
    --metric psnr ssim fid lpips ;
```
For DIV2K, degradation-aware prompts are used. Therefore, prompts file path depends on measurement.
```
python solve.py \
    --workdir experiments/0722_div2k_sr_avgpool \
    --measurement_path datasets/prepared/DIV2K_train800/measurement/sr_avgpool_x12 \
    --num_samples 800 \
    --prompt_file datasets/prepared/DIV2K_train800/prompts_sr_avgpool_x12.txt \
    --task sr_avgpool \
    --deg_scale 12 \
    --efficient_memory;
```

### Examples

You can quickly check the results using the following examples.

**Example 1. Super-resolution x 12 (avg-pool) / Dog**
```
python solve.py \
    --img_size 768 \
    --img_path samples/afhq_example.jpg \
    --prompt "a photo of a closed face of a dog" \
    --task sr_avgpool \
    --deg_scale 12 \
    --efficient_memory;
```

**Example 2. Super-resolution x 12 (bicubic) / Animal**
```
python solve.py \
    --img_size 768 \
    --img_path samples/div2k_example.png \
    --prompt "a high quality photo of animal, bush, close-up, fox, grass, green, greenery, hide, panda, red, red panda, stare" \
    --task sr_bicubic \
    --deg_scale 12 \
    --efficient_memory;
```
> The prompt (after "a high quality photo of") is extracted by DAPE from the given measurement.

**Example 3. Motion Deblur / Human**
```
python solve.py \
    --img_size 768 \
    --img_path samples/ffhq_example.png \
    --prompt "a photo of a closed face" \
    --task deblur_motion \
    --deg_scale 61 \
    --efficient_memory;
```


For each task, expected results are
![expect](assets/expected.jpg)


### Arbitrary size problem
You can solve inverse problems for rectangular-shaped images. 

```bash
python solve_arbitrary.py \
    --imgH 768 \
    --imgW 1152 \
    --img_path samples/div2k_example.png \
    --prompt "a high quality photo of animal, bush, close-up, fox, grass, green, greenery, hide, panda, red, red panda, stare" \
    --task deblur_motion \
    --deg_scale 61 \
    --efficient_memory;
```

Measurement            |  Reconstruction
:-------------------------:|:-------------------------:
![](assets/rect_input.png)  |  ![](assets/rect_output.png)

## How to choose task and solver

You can freely change the task and solver using the following arguments:
- `task` : sr_avgpool / sr_bicubic / deblur_gauss / deblur_motion
- `method` : psld / flowchef / flowdps

If you want to change the amount of degradation, change `deg_scale`. For SR tasks, it refers to the downscaling factor, and for deblurring tasks, it refers to the kernel size. 

## Efficient inference

If you use `--efficient_memory`, the text encoder will pre-compute text embeddings and be removed from the GPU.

This allows us to solve inverse problem with a single GPU with VRAM of 24GB.
