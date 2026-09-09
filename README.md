# Scribble-Supervised Binary Image Segmentation

[![CI](https://github.com/KashishMendiratta/Binary_Image_Segmentation/actions/workflows/ci.yml/badge.svg)](https://github.com/KashishMendiratta/Binary_Image_Segmentation/actions/workflows/ci.yml)

Binary image segmentation from sparse scribble annotations, comparing classical
methods with a compact U-Net trained from scratch and combining their strengths
through ensembling.

Developed as a course project for the Machine Learning Core Lecture at Saarland
University (Summer 2025).

**Best result: 75.8% mIoU** using a GrabCut + Tiny U-Net ensemble with four-way
test-time augmentation, compared with 59.9% for the pixel-wise KNN baseline.

## System overview

```text
sparse foreground/background scribbles
                  |
      classical models + Tiny U-Net
                  |
       test-time augmentation
                  |
       GrabCut/U-Net ensemble
                  |
          binary segmentation
```

## Results

| Method | Train/validation mIoU | CV score | Notes |
|---|:---:|:---:|---|
| Baseline KNN (k=3) | 59.9% | - | Pixel-wise, no tuning |
| Segment-aware KNN (k=9) | 60.2% | 57.3% | SLIC superpixels and Lab/spatial features |
| Random Forest | - | 62.4% | Used for pseudo-labelling |
| Random Walk | - | 45.2% | Weakest method under sparse supervision |
| GrabCut | - | 74.7% | Strongest classical baseline |
| Tiny U-Net | ~72% | - | Best individual CNN; no pretrained weights |
| **GrabCut + U-Net** | **75.8%** | - | **Final submission**, four-way TTA |

The complete methodology, ablations, and analysis are available in the
[project report](report/main.pdf).

## Key design decisions

- **Scribble-only supervision:** Thresholds and confidence settings were chosen
  using labelled scribble pixels rather than hidden ground-truth masks.
- **No pretrained weights:** The compact U-Net was trained from scratch to meet
  the project constraints.
- **Cross-validated model selection:** A 30-trial, three-fold search compared
  KNN, Random Forest, GrabCut, and Random Walk configurations.
- **Evidence-based complexity:** DenseCRF was evaluated but excluded because
  its marginal improvement did not justify the additional runtime.

## Repository structure

```text
.
├── challenge.py            # Training, tuning, ensembling, and inference
├── util.py                 # Data I/O, models, metrics, and visualisation
├── eval_unet_scribbles.py  # U-Net evaluation and threshold search
├── tests/                  # Unit tests for metrics and mask processing
├── report/main.pdf         # Full methodology and analysis
├── tiny_unet.pt            # Small reference checkpoint
└── requirements.txt        # Reproducible Python dependencies
```

## Installation

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

DenseCRF was used only for an ablation and is not required for the final model.
Install `pydensecrf` separately to reproduce that experiment.

## Data

The course dataset is not redistributed in this repository. Place an authorized
copy under `dataset/` using this layout:

```text
dataset/
├── train/
│   ├── images/
│   ├── scribbles/
│   └── ground_truth/
└── test1/
    ├── images/
    └── scribbles/
```

Scribble masks use `0` for labelled background, `1` for labelled foreground,
and `255` for unlabelled pixels. Generated predictions, logs, and experiment
artifacts are intentionally excluded from version control.

## Usage

Run model selection, U-Net training, ensembling, and inference:

```bash
python challenge.py \
    --data_root dataset \
    --trials 30 --folds 3 \
    --use_unet --unet_epochs 12 --unet_bs 2 \
    --final_tta 4way --ensemble avg_all
```

Evaluate the reference U-Net checkpoint:

```bash
python eval_unet_scribbles.py \
    --model tiny_unet.pt \
    --root dataset/train \
    --tta hflip
```

## Known limitation

The Tiny U-Net was trained without positive-class loss weighting due to the
CPU/MPS environment used for the project. This may bias predictions toward the
background class; the ensemble partially offsets that limitation.

## Author

Kashish Mendiratta
