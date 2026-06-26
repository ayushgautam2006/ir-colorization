"""
Reads downloaded band .tif files for each scene, combines Red/Green/Blue into
an RGB image, normalizes the thermal/IR band, and saves aligned IR-RGB pairs
as .npy and .png for inspection.
"""

import os
import numpy as np
import rasterio
from PIL import Image

RAW_DIR = "data/raw"
PROCESSED_DIR = "data/processed"


def normalize(band, lower_pct=2, upper_pct=98):
    """Normalize a band to 0-255 using percentile clipping (handles outliers)."""
    band = band.astype(np.float32)
    lo, hi = np.percentile(band, [lower_pct, upper_pct])
    band = np.clip(band, lo, hi)
    band = (band - lo) / (hi - lo + 1e-6) * 255.0
    return band.astype(np.uint8)


def load_band(path):
    with rasterio.open(path) as src:
        return src.read(1)


def process_scene(scene_dir, scene_index):
    red_path = os.path.join(scene_dir, "red.tif")
    green_path = os.path.join(scene_dir, "green.tif")
    blue_path = os.path.join(scene_dir, "blue.tif")
    thermal_path = os.path.join(scene_dir, "thermal.tif")

    if not all(os.path.exists(p) for p in [red_path, green_path, blue_path, thermal_path]):
        print(f"  [skip] scene_{scene_index}: missing one or more band files.")
        return

    red = normalize(load_band(red_path))
    green = normalize(load_band(green_path))
    blue = normalize(load_band(blue_path))
    ir = normalize(load_band(thermal_path))

    rgb = np.stack([red, green, blue], axis=-1)  # H x W x 3

    out_dir = os.path.join(PROCESSED_DIR, f"scene_{scene_index}")
    os.makedirs(out_dir, exist_ok=True)

    # Save as .npy for training pipeline
    np.save(os.path.join(out_dir, "rgb.npy"), rgb)
    np.save(os.path.join(out_dir, "ir.npy"), ir)

    # Save as .png for quick visual inspection
    Image.fromarray(rgb).save(os.path.join(out_dir, "rgb_preview.png"))
    Image.fromarray(ir).save(os.path.join(out_dir, "ir_preview.png"))

    print(f"  [ok] scene_{scene_index}: RGB shape {rgb.shape}, IR shape {ir.shape}")


def main():
    os.makedirs(PROCESSED_DIR, exist_ok=True)
    scene_dirs = sorted(
        [d for d in os.listdir(RAW_DIR) if d.startswith("scene_")]
    )

    for d in scene_dirs:
        idx = d.split("_")[1]
        process_scene(os.path.join(RAW_DIR, d), idx)

    print("\nDone. Processed pairs saved under:", PROCESSED_DIR)


if __name__ == "__main__":
    main()