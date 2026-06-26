"""
Gradio demo for IR -> RGB satellite image colorization.
Upload a grayscale IR image and see the model's colorized RGB output.
"""

import gradio as gr
import torch
import numpy as np
from PIL import Image

from models.generator import UNetGenerator

# ---- CONFIG ----
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CHECKPOINT_PATH = "checkpoints/generator_epoch69.pth"   # <-- update to your actual final epoch
IMAGE_SIZE = 256

# ---- LOAD MODEL ----
print(f"Loading model on {DEVICE}...")
generator = UNetGenerator().to(DEVICE)
generator.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=DEVICE))
generator.eval()
print("Model loaded successfully.")


def colorize(ir_image):
    if ir_image is None:
        return None

    # Preprocess: grayscale, resize, normalize to [-1, 1]
    ir = ir_image.convert("L").resize((IMAGE_SIZE, IMAGE_SIZE))
    ir_arr = np.array(ir).astype(np.float32) / 127.5 - 1.0
    ir_tensor = torch.from_numpy(ir_arr).unsqueeze(0).unsqueeze(0).to(DEVICE)  # [1, 1, H, W]

    # Run generator
    with torch.no_grad():
        rgb_fake = generator(ir_tensor)

    # Postprocess: [-1,1] -> [0,255] uint8
    rgb_fake = rgb_fake.squeeze(0).permute(1, 2, 0).cpu().numpy()
    rgb_fake = (rgb_fake + 1) / 2
    rgb_fake = np.clip(rgb_fake * 255, 0, 255).astype(np.uint8)

    return Image.fromarray(rgb_fake)


demo = gr.Interface(
    fn=colorize,
    inputs=gr.Image(type="pil", label="Upload IR (Infrared) Satellite Image"),
    outputs=gr.Image(type="pil", label="Generated Colorized RGB Output"),
    title="IR Satellite Image Enhancement & Colorization",
    description=(
        "Upload a grayscale infrared satellite image to generate a predicted RGB colorization. "
        "Built with a Pix2Pix conditional GAN trained on Landsat 8/9 thermal and RGB band pairs."
    ),
    examples=None,  # you can add example file paths here later if you want preset demo images
)

if __name__ == "__main__":
    demo.launch()