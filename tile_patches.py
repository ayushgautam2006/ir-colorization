"""
Slices full-scene IR/RGB .npy pairs into smaller paired patches (e.g. 256x256)
for training. Filters out mostly-empty/no-data tiles and splits into
train/val/test sets.
"""

import os
import numpy as np
from PIL import Image

PROCESSED_DIR = "data/processed"
TILES_DIR = "data/tiles"
TILE_SIZE = 256
STRIDE = 256          # set smaller than TILE_SIZE for overlapping tiles
BLACK_THRESHOLD = 0.6  # skip tile if more than 60% of pixels are near-zero
TRAIN_SPLIT, VAL_SPLIT = 0.8, 0.1  # remainder goes to test


def is_mostly_empty(arr, threshold=BLACK_THRESHOLD):
    """Check if an array is mostly black/zero (no-data)."""
    if arr.ndim == 3:
        gray = arr.mean(axis=-1)
    else:
        gray = arr
    empty_ratio = np.mean(gray < 5)
    return empty_ratio > threshold


def tile_scene(scene_dir, scene_name, all_tiles):
    rgb_path = os.path.join(scene_dir, "rgb.npy")
    ir_path = os.path.join(scene_dir, "ir.npy")

    if not (os.path.exists(rgb_path) and os.path.exists(ir_path)):
        print(f"  [skip] {scene_name}: missing rgb.npy or ir.npy")
        return

    rgb = np.load(rgb_path)
    ir = np.load(ir_path)

    h, w = ir.shape[:2]
    count = 0

    for y in range(0, h - TILE_SIZE + 1, STRIDE):
        for x in range(0, w - TILE_SIZE + 1, STRIDE):
            rgb_tile = rgb[y:y+TILE_SIZE, x:x+TILE_SIZE]
            ir_tile = ir[y:y+TILE_SIZE, x:x+TILE_SIZE]

            if is_mostly_empty(rgb_tile) or is_mostly_empty(ir_tile):
                continue

            tile_id = f"{scene_name}_tile_{count:04d}"
            all_tiles.append((tile_id, rgb_tile, ir_tile))
            count += 1

    print(f"  [ok] {scene_name}: {count} valid tiles extracted")


def save_split(tiles, split_name):
    rgb_dir = os.path.join(TILES_DIR, split_name, "rgb")
    ir_dir = os.path.join(TILES_DIR, split_name, "ir")
    os.makedirs(rgb_dir, exist_ok=True)
    os.makedirs(ir_dir, exist_ok=True)

    for tile_id, rgb_tile, ir_tile in tiles:
        Image.fromarray(rgb_tile).save(os.path.join(rgb_dir, f"{tile_id}.png"))
        Image.fromarray(ir_tile).save(os.path.join(ir_dir, f"{tile_id}.png"))

    print(f"  Saved {len(tiles)} tiles to {split_name}/")


def main():
    scene_dirs = sorted(
        [d for d in os.listdir(PROCESSED_DIR) if d.startswith("scene_")]
    )

    all_tiles = []
    for scene_name in scene_dirs:
        scene_path = os.path.join(PROCESSED_DIR, scene_name)
        tile_scene(scene_path, scene_name, all_tiles)

    print(f"\nTotal tiles extracted: {len(all_tiles)}")

    # Shuffle and split
    np.random.seed(42)
    np.random.shuffle(all_tiles)

    n = len(all_tiles)
    n_train = int(n * TRAIN_SPLIT)
    n_val = int(n * VAL_SPLIT)

    train_tiles = all_tiles[:n_train]
    val_tiles = all_tiles[n_train:n_train + n_val]
    test_tiles = all_tiles[n_train + n_val:]

    print(f"\nSplit -> train: {len(train_tiles)}, val: {len(val_tiles)}, test: {len(test_tiles)}")

    save_split(train_tiles, "train")
    save_split(val_tiles, "val")
    save_split(test_tiles, "test")

    print("\nDone. Tiles saved under:", TILES_DIR)


if __name__ == "__main__":
    main()