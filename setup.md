# Setup Guide — BRISC 2025 Brain Tumor Multitask Project

This guide covers how to set up the environment for all five model notebooks, covering both **Kaggle** and **Google Colab** platforms.

---

## Table of Contents

- [Platform Overview](#platform-overview)
- [Option A: Kaggle Setup](#option-a-kaggle-setup)
- [Option B: Google Colab Setup](#option-b-google-colab-setup)
- [Option C: Local Setup](#option-c-local-setup)
- [Dataset Setup](#dataset-setup)
- [Notebook-Specific Setup](#notebook-specific-setup)
- [Troubleshooting](#troubleshooting)

---

## Platform Overview

| Notebook                                      | Recommended Platform   | GPU Required  |
| --------------------------------------------- | ---------------------- | ------------- |
| `468-deeplabv3-final.ipynb`                   | Kaggle (CPU or GPU)    | No (CPU slow) |
| `cse468project.ipynb`                         | Kaggle (CPU or GPU)    | Recommended   |
| `468-updated-project.ipynb`                   | Kaggle (GPU T4)        | Yes           |
| `BRISC_Custom_UNet_Multitask_Colab_468.ipynb` | Google Colab (T4)      | Yes           |
| `_brain_multitask_colab_.ipynb`               | Google Colab (T4/A100) | Yes           |

---

## Option A: Kaggle Setup

### Step 1 — Create a Kaggle Account & Notebook

1. Go to [https://www.kaggle.com](https://www.kaggle.com) and sign in.
2. Click **"Create"** → **"New Notebook"**.
3. Upload the `.ipynb` file or paste the code cells.

### Step 2 — Enable GPU Accelerator

1. In the Kaggle notebook editor, go to **Settings** (right panel).
2. Under **Accelerator**, select **GPU T4 x2** or **GPU P100**.
3. Click **Save**.

### Step 3 — Add the BRISC 2025 Dataset

1. In the right sidebar, click **"Add data"** → **"Search Datasets"**.
2. Search for `brisc2025` and select **"BRISC 2025"** by `briscdataset`.
3. Click **"Add"**. The dataset will be mounted at `/kaggle/input/brisc2025/`.

Alternatively, the notebooks use `kagglehub` to download automatically:

```python
import kagglehub
path = kagglehub.dataset_download("briscdataset/brisc2025")
```

### Step 4 — Install Missing Packages

Most packages are pre-installed on Kaggle. For notebooks that need extra packages, add a cell at the top:

```python
# For Models 3 (LinkNet):
import subprocess, sys
subprocess.check_call([sys.executable, "-m", "pip", "install", "-q",
                       "segmentation-models-pytorch", "albumentations"])

# For Model 4 (Custom U-Net Colab):
# This notebook was designed for Google Colab, not Kaggle.
```

### Step 5 — Run All Cells

Click **"Run All"** or use **Shift+Enter** to execute cells sequentially.

---

## Option B: Google Colab Setup

### Step 1 — Open Google Colab

Go to [https://colab.research.google.com](https://colab.research.google.com) and sign in with a Google account.

### Step 2 — Upload the Notebook

- Click **File → Upload notebook** and select your `.ipynb` file, **OR**
- Upload to Google Drive and open from there via **File → Open notebook → Google Drive**.

### Step 3 — Enable GPU Runtime

1. Click **Runtime → Change runtime type**.
2. Under **Hardware accelerator**, choose **GPU** (T4 is free; A100 requires Colab Pro).
3. Click **Save**.

### Step 4 — Mount Google Drive & Upload Dataset

The Colab notebooks (`BRISC_Custom_UNet_Multitask_Colab_468.ipynb` and `_brain_multitask_colab_.ipynb`) load data from Google Drive:

```python
from google.colab import drive
drive.mount("/content/drive")
```

**Upload the dataset ZIP to Google Drive:**

1. Download the BRISC 2025 dataset from [Kaggle](https://www.kaggle.com/datasets/briscdataset/brisc2025) as a ZIP file.
2. Upload `archive.zip` to your **Google Drive root** (`My Drive/`).
3. The notebook will extract it automatically:

```python
!unzip -q /content/drive/MyDrive/archive.zip -d /content/brisc2025
```

Alternatively, use `kagglehub` inside Colab (requires your Kaggle API credentials):

```python
# Set up Kaggle API credentials first
import os
os.environ["KAGGLE_USERNAME"] = "your_kaggle_username"
os.environ["KAGGLE_KEY"] = "your_kaggle_api_key"

import kagglehub
path = kagglehub.dataset_download("briscdataset/brisc2025")
```

### Step 5 — Install Dependencies

Each Colab notebook includes an installation cell at the top. Run it first:

```bash
# Model 4 (Custom U-Net):
!pip install -q albumentations umap-learn seaborn scikit-learn timm

# Model 5 (ResNet50 + DeepLabV3+):
!pip install -q kagglehub segmentation-models-pytorch seaborn
```

---

## Option C: Local Setup

### Prerequisites

- Python 3.9 – 3.12
- NVIDIA GPU with CUDA 11.8+ (strongly recommended)
- `conda` or `venv` virtual environment

### Step 1 — Create a Virtual Environment

**Using conda (recommended):**

```bash
conda create -n brisc2025 python=3.10 -y
conda activate brisc2025
```

**Using venv:**

```bash
python -m venv brisc_env
# Windows:
brisc_env\Scripts\activate
# Linux/macOS:
source brisc_env/bin/activate
```

### Step 2 — Install PyTorch with CUDA

Visit [https://pytorch.org/get-started/locally/](https://pytorch.org/get-started/locally/) to get the exact command for your CUDA version. Example for CUDA 11.8:

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

For CPU only:

```bash
pip install torch torchvision torchaudio
```

### Step 3 — Install All Project Dependencies

```bash
pip install -r requirements.txt
```

### Step 4 — Download the Dataset

**Option A — Via Kaggle CLI:**

```bash
pip install kaggle
# Place your kaggle.json API key in ~/.kaggle/kaggle.json
kaggle datasets download -d briscdataset/brisc2025
unzip brisc2025.zip -d ./data/
```

**Option B — Via kagglehub (in notebook):**

```python
import kagglehub
path = kagglehub.dataset_download("briscdataset/brisc2025")
```

### Step 5 — Launch Jupyter

```bash
pip install jupyter
jupyter notebook
```

Open the desired `.ipynb` file.

---

## Dataset Setup

### Expected Directory Structure

After downloading and extracting, the dataset should look like:

```
brisc2025/
├── classification_task/
│   ├── train/
│   │   ├── glioma/          (1,147 images)
│   │   ├── meningioma/      (1,329 images)
│   │   ├── no_tumor/        (1,067 images)
│   │   └── pituitary/       (1,457 images)
│   └── test/
│       ├── glioma/          (254 images)
│       ├── meningioma/      (306 images)
│       ├── no_tumor/        (140 images)
│       └── pituitary/       (300 images)
├── segmentation_task/
│   ├── train/
│   │   ├── images/          (3,933 MRI scans)
│   │   └── masks/           (3,933 binary masks)
│   └── test/
│       ├── images/          (860 MRI scans)
│       └── masks/           (860 binary masks)
├── manifest.csv
├── manifest.json
└── README.md
```

> **Note:** The `no_tumor` class has no corresponding segmentation masks. All notebooks handle this by creating blank (all-zero) masks for those images automatically.

---

## Notebook-Specific Setup

### Model 1 — `468-deeplabv3-final.ipynb`

| Setting        | Value                                |
| -------------- | ------------------------------------ |
| Platform       | Kaggle                               |
| Input channels | 1 (grayscale)                        |
| Image size     | 128 × 128                            |
| Batch size     | 32                                   |
| Dataset path   | `/kaggle/input/brisc2025/brisc2025/` |

No extra packages needed (standard Kaggle environment).

---

### Model 2 — `cse468project.ipynb`

| Setting        | Value                                  |
| -------------- | -------------------------------------- |
| Platform       | Kaggle                                 |
| Input channels | 1 (grayscale)                          |
| Image size     | 256 × 256                              |
| Batch size     | 16                                     |
| CPU threads    | 4 (set via `torch.set_num_threads(4)`) |

No extra packages needed. The ENet architecture is fully custom — no external library required.

---

### Model 3 — `468-updated-project.ipynb`

| Setting        | Value                   |
| -------------- | ----------------------- |
| Platform       | Kaggle (CUDA preferred) |
| Input channels | 3 (RGB)                 |
| Image size     | 128 × 128               |
| Batch size     | 4                       |
| Epochs         | 8                       |

**Required extra packages (auto-installed):**

```bash
pip install segmentation-models-pytorch albumentations opencv-python-headless
```

---

### Model 4 — `BRISC_Custom_UNet_Multitask_Colab_468.ipynb`

| Setting              | Value                           |
| -------------------- | ------------------------------- |
| Platform             | Google Colab (Tesla T4)         |
| Input channels       | 3 (RGB)                         |
| Image size           | 224 × 224                       |
| Classification batch | 32                              |
| Segmentation batch   | 16                              |
| Max epochs           | 35 (early stopping, patience=7) |
| Dataset path         | `/content/brisc2025/brisc2025/` |

**Required packages:**

```bash
!pip install -q timm albumentations umap-learn seaborn
```

---

### Model 5 — `_brain_multitask_colab_.ipynb`

| Setting        | Value               |
| -------------- | ------------------- |
| Platform       | Google Colab (CUDA) |
| Input channels | 3 (RGB)             |
| Image size     | 224 × 224           |
| Batch size     | 16                  |
| Epochs         | 5                   |

**Required packages:**

```bash
!pip install -q kagglehub segmentation-models-pytorch seaborn
```

---

## Troubleshooting

### `FileNotFoundError: classification train folder not found`

**Cause:** The dataset has a nested `brisc2025/brisc2025/` directory.  
**Fix:** All notebooks auto-detect this. If running manually:

```python
import os
if os.path.exists(os.path.join(path, "brisc2025")):
    base_path = os.path.join(path, "brisc2025")
else:
    base_path = path
```

---

### `CUDA out of memory`

**Fix options:**

1. Reduce `BATCH_SIZE` (try halving it).
2. Reduce `IMG_SIZE` (e.g., 224 → 128).
3. Enable gradient accumulation.
4. Use `torch.cuda.empty_cache()` between batches.
5. Switch to mixed precision (already used in Models 4 and 5).

---

### `ModuleNotFoundError: No module named 'segmentation_models_pytorch'`

**Fix:**

```bash
pip install segmentation-models-pytorch
```

---

### `ModuleNotFoundError: No module named 'albumentations'`

**Fix:**

```bash
pip install albumentations
```

---

### `FutureWarning: torch.cuda.amp.GradScaler is deprecated`

**Fix (PyTorch ≥ 2.1):** Replace:

```python
# Old:
scaler = torch.cuda.amp.GradScaler()
# New:
scaler = torch.amp.GradScaler("cuda")
```

This is a warning only — code still functions correctly.

---

### Masks not found / all-zero masks

**Cause:** `no_tumor` images have no masks in the dataset by design.  
**Fix:** All notebooks handle this with:

```python
if pd.isna(row["mask_path"]):
    mask = Image.new("L", (IMG_SIZE, IMG_SIZE), 0)
```

This is expected behavior and does not indicate an error.

---

### Slow training on CPU (Kaggle)

**Fix:**

```python
torch.set_num_threads(4)
torch.set_num_interop_threads(1)
```

Or enable GPU in Kaggle Settings → Accelerator.

---

## Environment Verification

Run this cell to verify your environment:

```python
import torch, torchvision, sklearn, PIL, cv2, albumentations
import segmentation_models_pytorch as smp

print("PyTorch:", torch.__version__)
print("TorchVision:", torchvision.__version__)
print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
print("SMP version:", smp.__version__)
print("OpenCV:", cv2.__version__)
print("Albumentations:", albumentations.__version__)
```
