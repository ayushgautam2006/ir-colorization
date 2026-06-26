"""
Searches Microsoft Planetary Computer's STAC catalog for Landsat Collection 2
scenes and downloads the thermal (IR) and RGB bands - fully public, no AWS
account or credentials required.
"""

import os
import urllib.request
import planetary_computer as pc
from pystac_client import Client

# ---- CONFIG ----
BBOX = [86.0, 22.0, 87.0, 23.0]
DATE_RANGE = "2023-01-01/2023-06-01"
MAX_CLOUD_COVER = 20
MAX_SCENES = 5
OUTPUT_DIR = "data/raw"

# Use Landsat 8/9 OLI-TIRS Level-2 only (skip older Landsat 7 ETM which uses different band names)
COLLECTION = "landsat-c2-l2"

BANDS_TO_DOWNLOAD = {
    "lwir11": "thermal",   # OLI-TIRS thermal band
    "red": "red",
    "green": "green",
    "blue": "blue",
}


def search_scenes():
    catalog = Client.open("https://planetarycomputer.microsoft.com/api/stac/v1")
    search = catalog.search(
        collections=[COLLECTION],
        bbox=BBOX,
        datetime=DATE_RANGE,
        query={
            "eo:cloud_cover": {"lt": MAX_CLOUD_COVER},
            "platform": {"in": ["landsat-8", "landsat-9"]},  # skip Landsat 7 (different bands)
        },
        max_items=MAX_SCENES,
    )
    items = list(search.items())
    print(f"Found {len(items)} scenes matching criteria.")
    return items


def download_scene_bands(item, scene_index):
    scene_dir = os.path.join(OUTPUT_DIR, f"scene_{scene_index}")
    os.makedirs(scene_dir, exist_ok=True)

    print(f"\nAvailable assets for scene {scene_index}: {list(item.assets.keys())}")

    # Sign the item - this generates temporary authenticated URLs
    signed_item = pc.sign(item)

    for asset_key, label in BANDS_TO_DOWNLOAD.items():
        if asset_key not in signed_item.assets:
            print(f"  [skip] '{asset_key}' not found in this scene's assets.")
            continue

        url = signed_item.assets[asset_key].href
        out_path = os.path.join(scene_dir, f"{label}.tif")

        try:
            print(f"  Downloading {label} -> {out_path}")
            urllib.request.urlretrieve(url, out_path)

            if os.path.exists(out_path) and os.path.getsize(out_path) > 0:
                print(f"    ✓ saved {os.path.getsize(out_path)} bytes")
            else:
                print(f"    ✗ file missing or empty after download attempt")

        except Exception as e:
            print(f"  [error] failed to download {label}: {e}")

    return scene_dir


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    items = search_scenes()

    for idx, item in enumerate(items):
        download_scene_bands(item, idx)

    print("\nDone. Raw bands saved under:", OUTPUT_DIR)


if __name__ == "__main__":
    main()