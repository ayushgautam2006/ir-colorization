# 🛰️ IR Satellite Image Colorization

A **Pix2Pix conditional GAN** that colorizes grayscale infrared (thermal) satellite imagery into realistic RGB images. Trained on paired Landsat 8/9 thermal and RGB band data sourced from [Microsoft Planetary Computer](https://planetarycomputer.microsoft.com/).

---

## 🧠 How It Works

The model learns a mapping from a **single-channel thermal/IR image → 3-channel RGB image** using a Pix2Pix GAN architecture:

- **Generator** — A U-Net with 8 encoder and 8 decoder blocks, skip connections, and dropout for regularization. Takes `[B, 1, 256, 256]` IR images, outputs `[B, 3, 256, 256]` RGB images.
- **Discriminator** — A PatchGAN that classifies 30×30 local patches of concatenated IR+RGB pairs as real or fake, encouraging the generator to produce locally coherent texture.
- **Loss** — Adversarial (BCE) + pixel-wise L1 loss (`λ = 100`), which keeps outputs sharp and semantically aligned.

---

## 📁 Project Structure

```
ir-colorization/
├── models/
│   ├── generator.py        # UNetGenerator (8-level encoder-decoder with skip connections)
│   └── discriminator.py    # PatchGANDiscriminator (30×30 patch-level predictions)
├── data/
│   ├── raw/                # Downloaded .tif band files (per scene)
│   ├── processed/          # Combined rgb.npy + ir.npy per scene
│   └── tiles/              # 256×256 PNG patches split into train/val/test
├── checkpoints/            # Saved generator & discriminator weights per epoch
├── samples/                # Epoch-level sample grids (IR | Fake RGB | Real RGB)
├── results/                # Evaluation outputs (comparison images + metrics.txt)
├── download_data.py        # Step 1 – fetch Landsat scenes from Planetary Computer
├── combine_bands.py        # Step 2 – merge bands into aligned IR-RGB pairs
├── tile_patches.py         # Step 3 – slice scenes into 256×256 tiles & split
├── dataset.py              # PyTorch Dataset for loading paired IR/RGB tiles
├── train.py                # Training loop (Pix2Pix GAN)
├── evaluate.py             # Evaluation with PSNR & SSIM metrics
├── app.py                  # Gradio web demo for interactive inference
└── requirements.txt        # Python dependencies
```

---

## 🚀 Quickstart

### 1. Install Dependencies

```bash
pip install torch torchvision gradio scikit-image pystac-client planetary-computer boto3 rasterio numpy pillow
```

> **Note:** PyTorch and torchvision are not in `requirements.txt` — install the appropriate CUDA build from [pytorch.org](https://pytorch.org/get-started/locally/).

---

### 2. Download Satellite Data

```bash
python download_data.py
```

Searches Microsoft Planetary Computer for Landsat 8/9 scenes over a configurable bounding box and downloads the thermal (`lwir11`), red, green, and blue band `.tif` files into `data/raw/`.

**Configurable in `download_data.py`:**

| Parameter | Default | Description |
|---|---|---|
| `BBOX` | `[86.0, 22.0, 87.0, 23.0]` | Geographic bounding box (lon_min, lat_min, lon_max, lat_max) |
| `DATE_RANGE` | `2023-01-01/2023-06-01` | Scene search date range |
| `MAX_CLOUD_COVER` | `20` | Maximum cloud cover percentage |
| `MAX_SCENES` | `5` | Number of scenes to download |

---

### 3. Process & Align Bands

```bash
python combine_bands.py
```

Reads the raw `.tif` files, normalizes each band using percentile clipping (2nd–98th), stacks RGB channels, and saves aligned `rgb.npy` + `ir.npy` pairs (and PNG previews) to `data/processed/`.

---

### 4. Create Training Tiles

```bash
python tile_patches.py
```

Slices each scene into non-overlapping `256×256` patches, filters out mostly-black (no-data) tiles, then splits into **train / val / test** (80% / 10% / 10%) and saves as PNG files under `data/tiles/`.

---

### 5. Train the Model

```bash
python train.py
```

Trains the Pix2Pix GAN. Sample comparison grids are saved to `samples/` every 3 epochs, and model checkpoints are saved to `checkpoints/`.

**Key hyperparameters in `train.py`:**

| Parameter | Default | Description |
|---|---|---|
| `IMAGE_SIZE` | `256` | Input/output resolution |
| `BATCH_SIZE` | `8` | Batch size |
| `EPOCHS` | `30` | Total training epochs |
| `LR` | `2e-4` | Adam learning rate |
| `L1_LAMBDA` | `100` | L1 loss weight relative to GAN loss |
| `MAX_TRAIN_SAMPLES` | `1000` | Subset cap for fast iteration (`None` = all) |

---

### 6. Evaluate

```bash
python evaluate.py
```

Runs the trained generator on the test set and reports **PSNR** and **SSIM** metrics. Saves comparison images (IR | Predicted RGB | Ground Truth RGB) to `results/` and writes `results/metrics.txt`.

> Update `CHECKPOINT_PATH` in `evaluate.py` to point to your final epoch checkpoint.

---

### 7. Run the Demo

```bash
python app.py
```

Launches a [Gradio](https://gradio.app/) web interface where you can upload any grayscale IR satellite image and see the model's colorized RGB prediction in real time.

> Update `CHECKPOINT_PATH` in `app.py` to point to your trained checkpoint before running.

---

## 🏗️ Model Architecture

### Generator — U-Net

```
Input: [B, 1, 256, 256]  (grayscale IR)

Encoder (with skip connections):
  down1  →  [B,  64, 128, 128]
  down2  →  [B, 128,  64,  64]
  down3  →  [B, 256,  32,  32]
  down4  →  [B, 512,  16,  16]
  down5  →  [B, 512,   8,   8]
  down6  →  [B, 512,   4,   4]
  down7  →  [B, 512,   2,   2]
  bottleneck → [B, 512,   1,   1]

Decoder (with skip connections + dropout on first 3 up-blocks):
  up1 → up7 (mirrors encoder, concatenating skip features)

Output: [B, 3, 256, 256]  (RGB, Tanh activation → [-1, 1])
```

### Discriminator — PatchGAN

```
Input: concatenated [IR | RGB] → [B, 4, 256, 256]
Output: [B, 1, 30, 30]  (patch-level real/fake predictions)
```

---

## 📊 Training Outputs

| File | Description |
|---|---|
| `samples/epoch_XXX.png` | 3-row grid: IR / Fake RGB / Real RGB (4 examples per row) |
| `checkpoints/generator_epochXX.pth` | Generator weights |
| `checkpoints/discriminator_epochXX.pth` | Discriminator weights |
| `results/sample_XX.png` | Test set comparison triptychs |
| `results/metrics.txt` | Average PSNR and SSIM on the test set |

---

## 🛠️ Requirements

- Python 3.8+
- PyTorch (with CUDA recommended)
- `torchvision`
- `gradio`
- `scikit-image`
- `rasterio`
- `pystac-client`
- `planetary-computer`
- `boto3`
- `numpy`
- `pillow`

---

## 📄 Data Source

Satellite imagery is sourced from **Landsat Collection 2 Level-2** via the [Microsoft Planetary Computer STAC API](https://planetarycomputer.microsoft.com/). No AWS account or credentials are required — data is publicly accessible.

- **Sensor:** Landsat 8 / Landsat 9 OLI-TIRS
- **Thermal band:** `lwir11` (Band 10, ~10.9 µm)
- **RGB bands:** `red` (Band 4), `green` (Band 3), `blue` (Band 2)
