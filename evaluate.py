"""
Evaluates the trained generator on the test set using PSNR and SSIM.
Saves a grid of sample comparisons for visual inspection.
"""

import os
import torch
import numpy as np
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim

from dataset import IRColorizationDataset
from models.generator import UNetGenerator

# ---- CONFIG ----
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CHECKPOINT_PATH = "checkpoints/generator_epoch30.pth"   # change if your last epoch differs
IMAGE_SIZE = 256
BATCH_SIZE = 8
RESULTS_DIR = "results"

os.makedirs(RESULTS_DIR, exist_ok=True)


def denormalize(tensor):
    """[-1, 1] -> [0, 1]"""
    return (tensor + 1) / 2


def to_numpy_img(tensor):
    """C,H,W tensor in [0,1] -> H,W,C numpy in [0,255] uint8"""
    img = tensor.permute(1, 2, 0).cpu().numpy()
    img = np.clip(img * 255.0, 0, 255).astype(np.uint8)
    return img


def main():
    print(f"Using device: {DEVICE}")

    generator = UNetGenerator().to(DEVICE)
    generator.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=DEVICE))
    generator.eval()

    test_dataset = IRColorizationDataset("data/tiles", split="test", image_size=IMAGE_SIZE)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

    print(f"Test samples: {len(test_dataset)}")

    psnr_scores = []
    ssim_scores = []

    sample_count = 0
    with torch.no_grad():
        for batch in test_loader:
            ir = batch["ir"].to(DEVICE)
            rgb_real = batch["rgb"].to(DEVICE)
            rgb_fake = generator(ir)

            rgb_real_np = denormalize(rgb_real)
            rgb_fake_np = denormalize(rgb_fake)

            for i in range(rgb_real.size(0)):
                real_img = to_numpy_img(rgb_real_np[i])
                fake_img = to_numpy_img(rgb_fake_np[i])

                p = psnr(real_img, fake_img, data_range=255)
                s = ssim(real_img, fake_img, data_range=255, channel_axis=2)

                psnr_scores.append(p)
                ssim_scores.append(s)

                # save first 12 comparison images for visual inspection
                if sample_count < 12:
                    ir_rgb = ir[i].repeat(3, 1, 1)
                    comparison = torch.stack([
                        denormalize(ir_rgb),
                        rgb_fake_np[i],
                        rgb_real_np[i]
                    ])
                    save_image(comparison, os.path.join(RESULTS_DIR, f"sample_{sample_count:02d}.png"), nrow=3)
                sample_count += 1

    avg_psnr = np.mean(psnr_scores)
    avg_ssim = np.mean(ssim_scores)

    print(f"\n=== Evaluation Results (on {len(test_dataset)} test images) ===")
    print(f"Average PSNR: {avg_psnr:.2f} dB")
    print(f"Average SSIM: {avg_ssim:.4f}")
    print(f"\nSample comparisons saved to: {RESULTS_DIR}/")

    # save metrics to file too
    with open(os.path.join(RESULTS_DIR, "metrics.txt"), "w") as f:
        f.write(f"Average PSNR: {avg_psnr:.2f} dB\n")
        f.write(f"Average SSIM: {avg_ssim:.4f}\n")


if __name__ == "__main__":
    main()