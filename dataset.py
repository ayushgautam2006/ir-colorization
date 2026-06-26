"""
PyTorch Dataset for loading paired IR-RGB tiles for training.
"""

import os
from PIL import Image
import numpy as np
import torch
from torch.utils.data import Dataset


class IRColorizationDataset(Dataset):
    def __init__(self, tiles_dir, split="train", image_size=256):
        self.rgb_dir = os.path.join(tiles_dir, split, "rgb")
        self.ir_dir = os.path.join(tiles_dir, split, "ir")
        self.image_size = image_size

        self.filenames = sorted(os.listdir(self.rgb_dir))
        assert len(self.filenames) > 0, f"No files found in {self.rgb_dir}"

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        fname = self.filenames[idx]

        rgb = Image.open(os.path.join(self.rgb_dir, fname)).convert("RGB").resize((self.image_size, self.image_size))
        ir = Image.open(os.path.join(self.ir_dir, fname)).convert("L").resize((self.image_size, self.image_size))

        rgb = np.array(rgb).astype(np.float32) / 127.5 - 1.0
        ir = np.array(ir).astype(np.float32) / 127.5 - 1.0

        rgb = torch.from_numpy(rgb).permute(2, 0, 1)
        ir = torch.from_numpy(ir).unsqueeze(0)

        return {"ir": ir, "rgb": rgb, "filename": fname}


if __name__ == "__main__":
    # quick test
    dataset = IRColorizationDataset("data/tiles", split="train")
    print(f"Dataset size: {len(dataset)}")
    sample = dataset[0]
    print(f"IR shape: {sample['ir'].shape}, RGB shape: {sample['rgb'].shape}")
    print(f"IR range: [{sample['ir'].min():.2f}, {sample['ir'].max():.2f}]")
    print(f"RGB range: [{sample['rgb'].min():.2f}, {sample['rgb'].max():.2f}]")