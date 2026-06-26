"""
Training script for Pix2Pix IR -> RGB colorization.
Optimized for fast iteration: 128x128 resolution, reduced epochs, GPU.
"""

import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torchvision.utils import save_image

from dataset import IRColorizationDataset
from models.generator import UNetGenerator
from models.discriminator import PatchGANDiscriminator

# ---- CONFIG ----
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
IMAGE_SIZE = 256          # reverted - generator architecture requires this          # reduced from 256 -> ~4x faster per step
BATCH_SIZE = 8           # can increase since images are smaller now
EPOCHS = 30               # reduced from 100
LR = 2e-4
BETA1, BETA2 = 0.5, 0.999
L1_LAMBDA = 100
CHECKPOINT_DIR = "checkpoints"
SAMPLE_DIR = "samples"
SAVE_EVERY = 3
MAX_TRAIN_SAMPLES = 1000  # subset for faster epochs; set to None to use all 2509

os.makedirs(CHECKPOINT_DIR, exist_ok=True)
os.makedirs(SAMPLE_DIR, exist_ok=True)


def denormalize(tensor):
    return (tensor + 1) / 2


def save_sample(generator, val_loader, epoch):
    generator.eval()
    with torch.no_grad():
        batch = next(iter(val_loader))
        ir = batch["ir"].to(DEVICE)
        rgb_real = batch["rgb"].to(DEVICE)
        rgb_fake = generator(ir)

        ir_rgb = ir.repeat(1, 3, 1, 1)
        comparison = torch.cat([
            denormalize(ir_rgb[:4]),
            denormalize(rgb_fake[:4]),
            denormalize(rgb_real[:4])
        ], dim=0)

        save_image(comparison, os.path.join(SAMPLE_DIR, f"epoch_{epoch:03d}.png"), nrow=4)
    generator.train()


def main():
    print(f"Using device: {DEVICE}")

    train_dataset = IRColorizationDataset("data/tiles", split="train", image_size=IMAGE_SIZE)
    val_dataset = IRColorizationDataset("data/tiles", split="val", image_size=IMAGE_SIZE)

    if MAX_TRAIN_SAMPLES is not None and MAX_TRAIN_SAMPLES < len(train_dataset):
        train_dataset = Subset(train_dataset, range(MAX_TRAIN_SAMPLES))

    print(f"Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}")

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,
                               num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=True,
                             num_workers=2, pin_memory=True)

    generator = UNetGenerator().to(DEVICE)
    discriminator = PatchGANDiscriminator().to(DEVICE)

    opt_g = torch.optim.Adam(generator.parameters(), lr=LR, betas=(BETA1, BETA2))
    opt_d = torch.optim.Adam(discriminator.parameters(), lr=LR, betas=(BETA1, BETA2))

    criterion_gan = nn.BCEWithLogitsLoss()
    criterion_l1 = nn.L1Loss()

    for epoch in range(1, EPOCHS + 1):
        for i, batch in enumerate(train_loader):
            ir = batch["ir"].to(DEVICE)
            rgb_real = batch["rgb"].to(DEVICE)

            # ---- Train Discriminator ----
            opt_d.zero_grad()
            rgb_fake = generator(ir)

            pred_real = discriminator(ir, rgb_real)
            loss_d_real = criterion_gan(pred_real, torch.ones_like(pred_real))

            pred_fake = discriminator(ir, rgb_fake.detach())
            loss_d_fake = criterion_gan(pred_fake, torch.zeros_like(pred_fake))

            loss_d = (loss_d_real + loss_d_fake) * 0.5
            loss_d.backward()
            opt_d.step()

            # ---- Train Generator ----
            opt_g.zero_grad()
            pred_fake = discriminator(ir, rgb_fake)
            loss_g_gan = criterion_gan(pred_fake, torch.ones_like(pred_fake))
            loss_g_l1 = criterion_l1(rgb_fake, rgb_real) * L1_LAMBDA

            loss_g = loss_g_gan + loss_g_l1
            loss_g.backward()
            opt_g.step()

            if i % 10 == 0:
                print(f"Epoch [{epoch}/{EPOCHS}] Step [{i}/{len(train_loader)}] "
                      f"D_loss: {loss_d.item():.4f} G_loss: {loss_g.item():.4f} "
                      f"(GAN: {loss_g_gan.item():.4f}, L1: {loss_g_l1.item():.4f})")

        if epoch % SAVE_EVERY == 0 or epoch == 1:
            save_sample(generator, val_loader, epoch)
            torch.save(generator.state_dict(), os.path.join(CHECKPOINT_DIR, f"generator_epoch{epoch}.pth"))
            torch.save(discriminator.state_dict(), os.path.join(CHECKPOINT_DIR, f"discriminator_epoch{epoch}.pth"))
            print(f"  [checkpoint saved at epoch {epoch}]")

    print("Training complete.")


if __name__ == "__main__":
    main()