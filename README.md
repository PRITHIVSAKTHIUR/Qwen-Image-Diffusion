# **Qwen Image HPC – Diffusion**

> **Qwen Image HPC** is a Gradio web application for text-to-image generation using the Qwen/Qwen-Image diffusion model. It features flexible image generation controls, streamlined UI, easy batching, example prompts, and robust backend support for high-performance computing environments with GPU acceleration.

| ![Image 1](https://cdn-uploads.huggingface.co/production/uploads/65bb837dbfb878f46c77de4c/rMMNVotSQlJLtPTZkZZKL.png) | ![Image 2](https://cdn-uploads.huggingface.co/production/uploads/65bb837dbfb878f46c77de4c/Hwr9Wl4lGtuA-45rs7zRB.png) |
|---|---|
| ![Image 3](https://cdn-uploads.huggingface.co/production/uploads/65bb837dbfb878f46c77de4c/rESKwWQGaVJcy4JCvj_R-.png) | ![Image 4](https://cdn-uploads.huggingface.co/production/uploads/65bb837dbfb878f46c77de4c/_MGPYRXyfsWY8tuMhM6B-.png) |

---

## Features

- **Prompt-Based Image Generation:** Enter descriptive prompts to create high-quality images.
- **Negative Prompts:** Optionally guide the model to avoid specific elements in your images.
- **Aspect Ratio Selection:** Choose from common aspect ratios (1:1, 16:9, 9:16, 4:3, 3:4) with automatic size updates.
- **Seed Control:** Fix or randomize seed for reproducible or varied outputs.
- **Advanced Controls:** Adjust image size, guidance scale, inference steps, and batch size (number of images).
- **Output Management:** Download results individually or as a ZIP archive.
- **Examples:** Predefined sample prompts for quick testing.
- **Local File Saving:** Each image is automatically saved with a unique name for reference.
- **Performance Tracking:** Displays seed used and generation time.
- **UI Customization:** Simple, compact theme ideal for production or demonstration use.

## App Structure

- **Backend:** Uses Diffusers' `DiffusionPipeline` to load "Qwen/Qwen-Image" with `bfloat16` on CUDA (or CPU fallback).
- **Frontend:** Built with Gradio Blocks, features prompt input, aspect ratio dropdown, advanced settings accordions, and result galleries.
- **Helper Functions:** Manage random seed, image saving, and ZIP packaging.

## Installation

1. **Clone the repo:**
   ```
   git clone https://github.com/PRITHIVSAKTHIUR/Qwen-Image-Diffusion.git
   cd Qwen-Image-Diffusion
   ```
2. **Install dependencies:**
   ```bash
   pip install git+https://github.com/huggingface/transformers.git
   pip install git+https://github.com/huggingface/diffusers.git
   pip install git+https://github.com/huggingface/accelerate.git
   pip install git+https://github.com/huggingface/peft.git
   pip install huggingface-hub sentencepiece safetensors gradio torch pillow numpy
   ```

## Usage

```bash
python app.py
```
The app will launch a Gradio web interface. Enter your prompt and generate images!

## Example Prompts

- **Realistic Still Life:** "Realistic still life photography style: A single, fresh apple, resting on a clean, soft-textured surface..."
- **Chinese Painting:** "一幅精致细腻的工笔画，画面中心是一株蓬勃生长的红色牡丹..."
- **Classroom Scene:** "A young girl wearing school uniform stands in a classroom, writing on a chalkboard..."
- **Hand-drawn Water Cycle:** "手绘风格的水循环示意图，整体画面呈现出一幅生动形象的水循环过程图解..."
- **Capybara Mascot:** "A capybara wearing a suit, holding a sign, that reads Hello World"

## Customization

- **Aspect Ratio Quick Set:** Easily adjust width and height sliders to match standard ratios.
- **Batch Generation:** Create 1–5 images per prompt and package output as a ZIP.
- **UI Tweaks:** Modify the CSS and theme settings in `app.py` as needed.

## Repo

[https://github.com/PRITHIVSAKTHIUR/Qwen-Image-Diffusion](https://github.com/PRITHIVSAKTHIUR/Qwen-Image-Diffusion)

**Note:** Requires a machine with a supported NVIDIA GPU for best performance. For research and educational use.
