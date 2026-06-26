"""
Gradio demo for IR -> RGB colorization.
"""

import gradio as gr
import torch
import numpy as np
from PIL import Image

from models.generator import UNetGenerator

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CHECKPOINT_PATH = "checkpoints/generator_epoch30.pth"
IMAGE_SIZE = 256

generator = UNetGenerator().to(DEVICE)
generator.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=DEVICE))
generator.eval()


def colorize(ir_image):
    ir = ir_image.convert("L").resize((IMAGE_SIZE, IMAGE_SIZE))
    ir_arr = np.array(ir).astype(np.float32) / 127.5 - 1.0
    ir_tensor = torch.from_numpy(ir_arr).unsqueeze(0).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        rgb_fake = generator(ir_tensor)

    rgb_fake = (rgb_fake.squeeze(0).permute(1, 2, 0).cpu().numpy() + 1) / 2
    rgb_fake = np.clip(rgb_fake * 255, 0, 255).astype(np.uint8)

    return Image.fromarray(rgb_fake)


demo = gr.Interface(
    fn=colorize,
    inputs=gr.Image(type="pil", label="Upload IR Image"),
    outputs=gr.Image(type="pil", label="Colorized RGB Output"),
    title="IR Satellite Image Enhancement & Colorization",
    description="Upload a grayscale infrared satellite image to generate a colorized RGB translation."
)

if __name__ == "__main__":
    demo.launch()