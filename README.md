# Scribble-Supervised Binary Image Segmentation

Binary image segmentation from sparse scribble annotations, comparing classical
(KNN, Random Forest, GrabCut, Random Walk) and deep learning (Tiny U-Net, trained
from scratch — no pretrained weights) approaches, combined via ensembling.

Developed as a course project for the Machine Learning Core Lecture, Saarland
University (Summer 2025).

**Best result: 75.8% mIoU** (GrabCut + Tiny U-Net ensemble, 4-way TTA),
vs. 59.9% mIoU for the pixel-wise KNN baseline (k=3).

---

## Project Structure
```text
.
├── challenge.py            # Main pipeline: tuning, training, ensembling, inference
├── util.py                 # Dataset I/O, model implementations, evaluation, visualization
├── eval_unet_scribbles.py  # Standalone Tiny U-Net evaluation with TTA + threshold search
├── tiny_unet.pt  
└── results/
└── results_test2/
└── dataset/
    ├── train/
    │   ├── images/
    │   ├── scribbles/
    │   ├── ground_truth/
    │   └── predictions_s1 2_r320 284/    
    │   └── predictions_s1_r320/
    │   └── predictions_s1_r384/  
    │   └── predictions_s2_r320/ 
    │   └── predictions_s2_r384/ 
    └── test1/
    │   ├── images/
    │   ├── scribbles/
    │   └── predictions/    # created by challenge.py
    │   └── predictions_knn_baseline/
    │   └── predictions_refined/  
    │   └── predictions_unet_only/ 
    │   └── report_samples/ 
    |__ test2/
    │   ├── images/
    │   ├── scribbles/
    │   └── predictions/    # created by challenge.py
    │   └── predictions_knn_baseline/
    │   └── predictions_refined/  
    │   └── predictions_unet_only/ 
    │   └── report_samples/ 
```

---

## Installation

```bash
pip install numpy pillow matplotlib scikit-learn scikit-image opencv-python torch tqdm
# optional, for DenseCRF post-processing ablations:
pip install pydensecrf
```

---

## Usage

**Full pipeline** — classical model tuning (cross-validated), Tiny U-Net training,
ensembling, and inference on train + test splits:

```bash
python challenge.py \
    --data_root dataset \
    --trials 30 --folds 3 \
    --use_unet --unet_epochs 12 --unet_bs 2 \
    --final_tta 4way --ensemble avg_all
```

**Skip tuning**, reuse a saved config (`best_config.json`) or the top result
from a previous tuning log:

```bash
python challenge.py --trials 0 --use_unet --final_tta 4way --ensemble avg_all
```

**Standalone Tiny U-Net evaluation**, with test-time augmentation and
scribble-based threshold search:

```bash
python eval_unet_scribbles.py --model tiny_unet.pt --root dataset/train --tta hflip
```

**Post-hoc refinement only** (GrabCut + morphology + optional DenseCRF on
existing predictions):

```bash
python challenge.py --refine_only --refine_root dataset/test --refine_pred_dir predictions
```

---

## Method Summary

| Method                          | Train/Val mIoU | CV score | Notes |
|----------------------------------|:---:|:---:|---|
| Baseline KNN (k=3)               | 59.9% | – | pixel-wise, no tuning |
| KNN+ (segment-aware, k=9)        | 60.2% | 57.3% | SLIC superpixels + Lab/spatial features |
| Random Forest (teacher)          | – | 62.4% | used for pseudo-labeling, not final output |
| Random Walk                      | – | 45.2% | weakest method, poor robustness to sparse scribbles |
| GrabCut                          | – | 74.7% | strongest classical baseline |
| Tiny U-Net (from scratch)        | ~72% | – | best individual CNN; no pretrained weights |
| **GrabCut + U-Net ensemble**     | **75.8%** | – | **submitted result**, 4-way TTA |

Full methodology, ablations, and analysis in [`report/main.pdf`](report/main.pdf).

### Key design decisions
- **Scribble-only supervision**: all thresholds (binarization τ, RF confidence)
  were selected using scribble pixels only, never ground truth — keeping the
  pipeline honest to the sparse-label setting.
- **No pretrained weights**: Tiny U-Net (base channels 16, 3 encoder / 2 decoder
  blocks) was trained from scratch, per project constraints.
- **Cross-validated config search**: 30-trial, 3-fold CV over classical model
  families (KNN / Random Forest / GrabCut) before selecting the ensemble.
- **DenseCRF was evaluated but excluded** from the final submission — gains
  were marginal (scribble-accuracy 0.9995 → 1.0) relative to added runtime.

### Known limitation
Tiny U-Net was trained without `pos_weight` class rebalancing (for stability on
Mac CPU/MPS hardware without GPU acceleration), which may bias predictions
toward background. The ensemble partially offsets this — see report Limitations
section for discussion.

---

## Author
Kashish Mendiratta 
