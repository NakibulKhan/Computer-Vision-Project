# BRISC 2025 — Brain Tumor Multitask Learning

> **CSE 468 Project** | Joint Brain Tumor **Classification** + **Segmentation** on the BRISC 2025 Dataset  
> Five deep learning models trained and evaluated for simultaneous tumor type classification and lesion segmentation.

## Table of Contents

- [Project Overview](#project-overview)
- [Dataset](#dataset)
- [Models](#models)
- [Results Summary](#results-summary)
- [Project Structure](#project-structure)
- [Setup & Installation](#setup--installation)
- [Usage](#usage)
- [Training Details](#training-details)
- [Evaluation Metrics](#evaluation-metrics)
- [Team & References](#team--references)

---

## Project Overview

This project tackles a **multitask learning** problem on brain MRI scans: simultaneously predicting the **tumor type** (classification) and **localizing the tumor region** (segmentation). Five different model architectures with various backbones were trained and compared on the BRISC 2025 benchmark dataset.

### Task Definition

| Task                 | Type                    | Output                                                 |
| -------------------- | ----------------------- | ------------------------------------------------------ |
| Tumor Classification | Multi-class (4 classes) | Softmax over {glioma, meningioma, no_tumor, pituitary} |
| Tumor Segmentation   | Binary                  | Pixel-wise binary mask (tumor / background)            |

---

## Dataset

**BRISC 2025** (`briscdataset/brisc2025` on Kaggle)

| Split | Images | Masks |
| ----- | ------ | ----- |
| Train | 5,000  | 3,933 |
| Test  | 1,000  | 860   |

### Class Distribution (Training Set)

| Class      | Count |
| ---------- | ----- |
| Pituitary  | 1,457 |
| Meningioma | 1,329 |
| No Tumor   | 1,067 |
| Glioma     | 1,147 |

**Key notes:**

- `no_tumor` images have **no segmentation mask** (blank mask used during training).
- Images are brain MRI scans in axial, sagittal, and coronal planes (T1 weighted).
- Masks are binary pixel-wise annotations of the tumor region.
- Image IDs encode scan metadata: e.g., `brisc2025_train_00001_gl_ax_t1` → glioma, axial, T1.

---

## Models

Five architectures were trained, each implementing joint classification + segmentation heads.

### Model 1 — DeepLabV3 + MobileNetV3-Large

**Notebook:** `468-deeplabv3-final.ipynb`

- **Architecture:** Custom `DeepLabV3WithClassifier` wrapping `deeplabv3_mobilenet_v3_large`
- **Backbone:** MobileNetV3-Large (modified first conv for grayscale input)
- **Segmentation Head:** Standard DeepLabV3 ASPP head → 1-channel output
- **Classification Head:** `AdaptiveAvgPool2d` → `Linear(960, 4)`
- **Input:** Grayscale (1-channel), 128×128
- **Platform:** Kaggle (CPU)

### Model 2 — ENet (from scratch)

**Notebook:** `cse468project.ipynb`

- **Architecture:** Custom `ENetWithClassifier` — full from-scratch ENet implementation
- **Backbone:** ENet encoder (InitialBlock → Stage1 → Stage2 → Stage3) with MaxPool unpooling decoder
- **Features:** PReLU activations, dilated convolutions (d=2,4,8,16), asymmetric (5×1 + 1×5) convolutions, Dropout2d
- **Segmentation Head:** ConvTranspose2d → 1-channel
- **Classification Head:** `AdaptiveAvgPool2d(1)` → `Linear(128, 4)` from encoder bottleneck
- **Input:** Grayscale (1-channel), 256×256
- **Batch Size:** 16 | **Optimizer:** AdamW
- **Platform:** Kaggle (CPU)

### Model 3 — LinkNet + EfficientNet-B0

**Notebook:** `468-updated-project.ipynb`

- **Architecture:** `smp.Linknet(encoder_name="efficientnet-b0", encoder_weights="imagenet")`
- **Backbone:** EfficientNet-B0 (ImageNet pretrained)
- **Segmentation:** LinkNet decoder → 1-channel binary mask
- **Classification Head:** AuxParams `pooling=avg, dropout=0.30, classes=4`
- **Input:** RGB (3-channel), 128×128 | ImageNet normalization
- **Augmentation:** Albumentations — HorizontalFlip, ShiftScaleRotate
- **Split:** 70% train / 15% val / 15% test
- **Loss:** `0.7 × (0.5·BCE + 0.5·Dice) + 0.3 × CrossEntropy`
- **Optimizer:** AdamW(lr=2e-4, weight_decay=1e-4) | **Epochs:** 8 | **Batch Size:** 4
- **Platform:** Kaggle (CUDA)

### Model 4 — Custom U-Net Multitask

**Notebook:** `BRISC_Custom_UNet_Multitask_Colab_468.ipynb`

- **Architecture:** Custom `CustomUNetMultiTask` (from-scratch U-Net with skip connections)
- **Encoder:** 4-stage ConvBlock encoder (base=32 → 64 → 128 → 256 channels) + MaxPool
- **Bottleneck:** ConvBlock(256, 512)
- **Segmentation Decoder:** 4 upsampling stages with skip connections via ConvTranspose2d + ConvBlock
- **Classification Head:** Bottleneck → `AdaptiveAvgPool2d(1)` → FC(512→256, BN, ReLU, Dropout(0.45)) → Linear(256, 4)
- **Total Parameters:** 7.896M
- **Input:** RGB (3-channel), 224×224 | ImageNet normalization
- **Augmentation:** Albumentations — HorizontalFlip, ShiftScaleRotate, RandomBrightnessContrast, GaussianBlur, CoarseDropout
- **Loss:** `CrossEntropy(label_smoothing=0.08) + 0.5·BCE + 0.5·Dice`
- **Optimizer:** AdamW(lr=3e-4, weight_decay=1e-4) | **Scheduler:** ReduceLROnPlateau(factor=0.5, patience=3)
- **Early Stopping:** patience=7 | **Max Epochs:** 35 | **Batch Size (cls):** 32, **(seg):** 16
- **Mixed Precision:** `torch.cuda.amp.GradScaler` | **Grad clipping:** max_norm=1.0
- **Platform:** Google Colab (Tesla T4)
- **Model selection:** Best of `0.5 × F1 + 0.5 × Dice`

### Model 5 — ResNet50 + DeepLabV3+

**Notebook:** `_brain_multitask_colab_.ipynb`

- **Architecture:** Custom `MultiTaskModel` — dual-branch design
- **Classification Branch:** ResNet50 (ImageNet V2 pretrained) → `AdaptiveAvgPool2d` → `Linear(2048, 4)`
- **Segmentation Branch:** `smp.DeepLabV3Plus(encoder_name="resnet50", encoder_weights="imagenet", classes=1)`
- **Input:** RGB (3-channel), 224×224 | ImageNet normalization
- **Loss:** `CrossEntropyLoss (cls) + BCEWithLogitsLoss (seg)` (equal weight)
- **Optimizer:** AdamW(lr=1e-4, weight_decay=1e-4) | **Epochs:** 5 | **Batch Size:** 16
- **Mixed Precision:** `torch.cuda.amp.GradScaler`
- **Data Split:** 80% train / 20% val + official test set (1000 images)
- **Platform:** Google Colab (CUDA)

---

## Results Summary

### Classification Results (Validation / Test)

| Model                     | Backbone          | Accuracy            | Macro F1   |
| ------------------------- | ----------------- | ------------------- | ---------- |
| DeepLabV3 + MobileNetV3   | MobileNetV3-Large | 0.291 (1 ep, CPU)   | —          |
| ENet (from scratch)       | ENet encoder      | Limited epochs      | —          |
| LinkNet + EfficientNet-B0 | EfficientNet-B0   | Improving over 8 ep | —          |
| Custom U-Net Multitask    | Custom CNN        | **0.9477**          | **0.9478** |
| ResNet50 + DeepLabV3+     | ResNet50          | **0.9760**          | **0.9790** |

### Segmentation Results (Validation / Test)

| Model                     | Val IoU           | Val Dice   |
| ------------------------- | ----------------- | ---------- |
| DeepLabV3 + MobileNetV3   | 0.0000 (1 ep)     | —          |
| ENet (from scratch)       | Limited epochs    | —          |
| LinkNet + EfficientNet-B0 | Improving         | —          |
| Custom U-Net Multitask    | **0.6707**        | **0.7652** |
| ResNet50 + DeepLabV3+     | **0.7175** (test) | —          |

### Best Model — ResNet50 + DeepLabV3+ (Official Test Set)

```
Classification Report — Official Test Set (1000 images)
              precision    recall  f1-score   support

      glioma     0.9685    0.9685    0.9685       254
  meningioma     0.9551    0.9739    0.9644       306
    no_tumor     1.0000    1.0000    1.0000       140
   pituitary     0.9932    0.9733    0.9832       300

    accuracy                         0.9760      1000
   macro avg     0.9792    0.9789    0.9790      1000
weighted avg     0.9762    0.9760    0.9761      1000

Official Test Segmentation IoU: 0.7175
```

---

## Project Structure

```
468/
├── README.md                              ← This file
├── setup.md                               ← Environment setup guide
├── requirements.txt                       ← Python dependencies
│
├── 468-deeplabv3-final.ipynb             ← Model 1: DeepLabV3 + MobileNetV3
├── cse468project.ipynb                   ← Model 2: ENet (from scratch)
├── 468-updated-project.ipynb             ← Model 3: LinkNet + EfficientNet-B0
├── BRISC_Custom_UNet_Multitask_Colab_468.ipynb  ← Model 4: Custom U-Net
├── _brain_multitask_colab_.ipynb         ← Model 5: ResNet50 + DeepLabV3+
│
└── Final Report.docx                     ← Project report
```

---

## Setup & Installation

See **[setup.md](setup.md)** for full environment setup instructions.

**Quick start (Kaggle):**

```python
import kagglehub
path = kagglehub.dataset_download("briscdataset/brisc2025")
```

**Quick start (Google Colab):**

```python
from google.colab import drive
drive.mount("/content/drive")
!unzip /content/drive/MyDrive/archive.zip -d /content/brisc2025
```

---

## Training Details

### Loss Functions

All models use a combination of:

- **Segmentation:** `BCEWithLogitsLoss + DiceLoss` (usually 50/50 blend)
- **Classification:** `CrossEntropyLoss` (with optional class weights for imbalance)
- **Total:** Weighted combination `α·SegLoss + (1-α)·ClsLoss`

### Data Augmentation

| Transform                              | Models         |
| -------------------------------------- | -------------- |
| RandomHorizontalFlip                   | All            |
| RandomRotation (±10–15°)               | All            |
| ShiftScaleRotate                       | Models 3, 4, 5 |
| ColorJitter / RandomBrightnessContrast | Models 1, 2, 4 |
| AutoContrast / Contrast Stretch        | Models 1, 2    |
| CoarseDropout                          | Model 4        |
| GaussianBlur                           | Model 4        |

### Reproducibility

All notebooks set global seeds:

```python
SEED = 42
random.seed(SEED); np.random.seed(SEED)
torch.manual_seed(SEED); torch.cuda.manual_seed_all(SEED)
```

---

## Evaluation Metrics

| Metric                | Task           | Description                                                |
| --------------------- | -------------- | ---------------------------------------------------------- | --- | --- | --- | --- | --- | --- |
| **Accuracy**          | Classification | Fraction of correctly classified images                    |
| **Macro F1**          | Classification | Unweighted mean F1 across all 4 classes                    |
| **Balanced Accuracy** | Classification | Average recall across classes                              |
| **IoU (Jaccard)**     | Segmentation   | Intersection over Union of predicted vs. ground-truth mask |
| **Dice Score**        | Segmentation   | `2·                                                        | P∩T | / ( | P   | +   | T   | )`  |

---

## Team & References

**Course:** CSE 468  
**Dataset:** BRISC 2025 — [Kaggle: briscdataset/brisc2025](https://www.kaggle.com/datasets/briscdataset/brisc2025)

### Key Libraries

- [PyTorch](https://pytorch.org/) — Deep learning framework
- [segmentation-models-pytorch (SMP)](https://github.com/qubvel/segmentation_models.pytorch) — Pre-built segmentation architectures
- [Albumentations](https://albumentations.ai/) — Image augmentation
- [scikit-learn](https://scikit-learn.org/) — Metrics and data splitting
- [timm](https://github.com/huggingface/pytorch-image-models) — Pretrained vision models

### Architecture References

- **ENet:** Paszke et al., "ENet: A Deep Neural Network Architecture for Real-Time Semantic Segmentation" (2016)
- **DeepLabV3:** Chen et al., "Rethinking Atrous Convolution for Semantic Image Segmentation" (2017)
- **DeepLabV3+:** Chen et al., "Encoder-Decoder with Atrous Separable Convolution for Semantic Image Segmentation" (2018)
- **LinkNet:** Chaurasia & Culurciello, "LinkNet: Exploiting Encoder Representations for Efficient Semantic Segmentation" (2017)
- **EfficientNet:** Tan & Le, "EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks" (2019)
