#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Evaluate Tiny U-Net on a dataset with scribbles + ground-truth masks.

- Loads tiny_unet.pt (your trained checkpoint)
- Predicts prob_fg maps with TTA (hflip by default)
- Selects threshold τ via scribble-only grid search (per-image by default)
- Computes mIoU (bg+fg averaged) and scribble-accuracy
- (Optional) saves qualitative figures and a CSV with per-image metrics

Expected dataset layout (adjust with --images/--scribbles/--masks if needed):
dataset/
  train/
    images/*.jpg|*.png
    scribbles/*.png        # values in {0,1,255}
    ground_truth/*.png     # values in {0,1}
"""

import argparse, os, sys, json
from pathlib import Path

import numpy as np
from PIL import Image

import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

# Import your helpers from util.py
from util import TinyUNet, _to_tensor_img, threshold_with_scribbles

# ----------------------------
# Utilities
# ----------------------------
def load_image(path):
    return np.array(Image.open(path).convert("RGB"))

def load_mask(path):
    # preserves integers (0/1/255) if stored as palette/uint8 png
    return np.array(Image.open(path))

def predict_unet_prob(net, img_rgb, resize=320, tta="hflip", device="cpu"):
    def _forward(img):
        x = _to_tensor_img(img, resize).unsqueeze(0).to(device)  # 1,C,R,R
        with torch.no_grad():
            logit = net(x).squeeze(0).squeeze(0)                  # R,R
            prob  = torch.sigmoid(logit)[None,None]
            prob  = F.interpolate(prob, size=img.shape[:2], mode="bilinear", align_corners=False)
            return prob.squeeze().cpu().numpy().astype(np.float32)

    if tta == "none":
        return _forward(img_rgb)
    elif tta == "hflip":
        p1 = _forward(img_rgb)
        p2 = _forward(img_rgb[:, ::-1, :])[:, ::-1]
        return 0.5 * (p1 + p2)
    elif tta == "4way":
        p = []
        p.append(_forward(img_rgb))
        p.append(_forward(img_rgb[:, ::-1, :])[:, ::-1])
        p.append(_forward(img_rgb[::-1, :, :])[::-1, :])
        p.append(_forward(img_rgb[::-1, ::-1, :])[::-1, ::-1])
        return np.mean(p, axis=0)
    else:
        raise ValueError(f"Unknown TTA: {tta}")

def compute_iou(pred_bin, gt_bin):
    pred_bin = (pred_bin > 0).astype(np.uint8)
    gt_bin   = (gt_bin   > 0).astype(np.uint8)

    inter_fg = np.logical_and(pred_bin==1, gt_bin==1).sum()
    union_fg = np.logical_or (pred_bin==1, gt_bin==1).sum()
    iou_fg = inter_fg / (union_fg + 1e-6)

    inter_bg = np.logical_and(pred_bin==0, gt_bin==0).sum()
    union_bg = np.logical_or (pred_bin==0, gt_bin==0).sum()
    iou_bg = inter_bg / (union_bg + 1e-6)

    miou = 0.5 * (iou_fg + iou_bg)
    return iou_bg, iou_fg, miou

def compute_scribble_acc(pred_bin, scribble):
    L = (scribble != 255)
    if L.sum() == 0:
        return np.nan
    return (pred_bin[L] == scribble[L]).mean()

def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)
    return p

# ----------------------------
# Main
# ----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=str, default="tiny_unet.pt", help="Path to tiny_unet.pt")
    ap.add_argument("--root", type=str, default="dataset/train", help="Dataset root directory")
    ap.add_argument("--images", type=str, default="images", help="Subdir for images")
    ap.add_argument("--scribbles", type=str, default="scribbles", help="Subdir for scribbles")
    ap.add_argument("--masks", type=str, default="ground_truth", help="Subdir for ground-truth masks")
    ap.add_argument("--ext", type=str, default="jpg,png", help="Image extensions to consider (comma-separated)")
    ap.add_argument("--resize", type=int, default=320, help="Resize side for network")
    ap.add_argument("--tta", type=str, default="hflip", choices=["none","hflip","4way"], help="Test-time augmentation")
    ap.add_argument("--per_image_tau", action="store_true", help="Use per-image threshold from scribbles (default)")
    ap.add_argument("--global_tau", action="store_true", help="Use a single global threshold chosen by averaging per-image best τ")
    ap.add_argument("--tau_grid", type=str, default="0.35,0.40,0.45,0.50,0.55,0.60,0.65", help="Threshold grid for scribble search")
    ap.add_argument("--save_figs", type=str, default="", help="Directory to save qualitative figures (optional)")
    ap.add_argument("--save_csv",  type=str, default="", help="Path to save per-image metrics CSV (optional)")
    args = ap.parse_args()

    # Resolve flags
    if not args.global_tau:
        args.per_image_tau = True  # default behavior

    root = Path(args.root)
    img_dir = root / args.images
    scr_dir = root / args.scribbles
    gt_dir  = root / args.masks

    # Gather files
    exts = tuple(["."+e.strip() for e in args.ext.split(",") if e.strip()])
    imgs = sorted([p for p in img_dir.iterdir() if p.suffix.lower() in exts])
    scrs = [scr_dir / (p.stem + ".png") for p in imgs]    # scribbles as .png
    gts  = [gt_dir  / (p.stem + ".png") for p in imgs]    # gt as .png

    if len(imgs) == 0:
        print(f"[ERROR] No images found in {img_dir} with extensions {exts}.", file=sys.stderr)
        sys.exit(1)

    # Load model
    device = torch.device("cuda" if torch.cuda.is_available() else
                          ("mps" if torch.backends.mps.is_available() else "cpu"))
    net = TinyUNet().Net.to(device)
    net.load_state_dict(torch.load(args.model, map_location=device))
    net.eval()
    print(f"[INFO] Loaded model from {args.model} on device={device}")

    # Optional outputs
    if args.save_figs:
        ensure_dir(Path(args.save_figs))
    rows = []

    # If using global τ: first pass to get per-image best τ, then average
    tau_grid = np.array([float(x) for x in args.tau_grid.split(",")])
    global_tau = None
    per_image_tau_list = []

    # First pass: optionally collect per-image τ*
    if args.global_tau:
        for p_img, p_scr in zip(imgs, scrs):
            img = load_image(p_img)
            scr = load_mask(p_scr)
            prob = predict_unet_prob(net, img, resize=args.resize, tta=args.tta, device=device)
            tstar = threshold_with_scribbles(prob, scr, grid=tau_grid)
            per_image_tau_list.append(tstar)
        global_tau = float(np.mean(per_image_tau_list))
        print(f"[INFO] Global τ (mean of per-image τ*): {global_tau:.3f}")

    # Second pass: evaluate metrics (and save figs)
    all_iou_bg, all_iou_fg, all_miou, all_scrib = [], [], [], []
    for idx, (p_img, p_scr, p_gt) in enumerate(zip(imgs, scrs, gts)):
        if not p_scr.exists() or not p_gt.exists():
            # Skip if missing annotations
            continue

        img = load_image(p_img)
        scr = load_mask(p_scr)
        gt  = load_mask(p_gt)

        prob = predict_unet_prob(net, img, resize=args.resize, tta=args.tta, device=device)

        if args.per_image_tau:
            tau = threshold_with_scribbles(prob, scr, grid=tau_grid)
        else:
            tau = global_tau if global_tau is not None else 0.5  # fallback

        pred = (prob >= float(tau)).astype(np.uint8)

        iou_bg, iou_fg, miou = compute_iou(pred, gt)
        scrib_acc = compute_scribble_acc(pred, scr)

        all_iou_bg.append(iou_bg); all_iou_fg.append(iou_fg)
        all_miou.append(miou);     all_scrib.append(scrib_acc)

        rows.append({
            "image": p_img.name,
            "tau": float(tau),
            "iou_bg": float(iou_bg),
            "iou_fg": float(iou_fg),
            "mIoU": float(miou),
            "scribble_acc": float(scrib_acc) if scrib_acc==scrib_acc else None
        })

        # Optional: save qualitative figure
        if args.save_figs:
            plt.figure(figsize=(12,4))
            plt.subplot(1,4,1); plt.imshow(img); plt.title("Input"); plt.axis("off")
            plt.subplot(1,4,2); plt.imshow(scr, cmap="gray", vmin=0, vmax=255); plt.title("Scribbles"); plt.axis("off")
            plt.subplot(1,4,3); plt.imshow(pred, cmap="gray", vmin=0, vmax=1); plt.title(f"Pred τ={tau:.2f}"); plt.axis("off")
            plt.subplot(1,4,4); plt.imshow(gt, cmap="gray", vmin=0, vmax=1); plt.title("Ground Truth"); plt.axis("off")
            out_path = Path(args.save_figs) / f"{p_img.stem}_qual.png"
            plt.tight_layout(); plt.savefig(out_path, dpi=160); plt.close()

    # Aggregate
    mean_iou_bg = float(np.mean(all_iou_bg)) if all_iou_bg else float("nan")
    mean_iou_fg = float(np.mean(all_iou_fg)) if all_iou_fg else float("nan")
    mean_miou   = float(np.mean(all_miou))   if all_miou   else float("nan")
    mean_scrib  = float(np.mean(all_scrib))  if all_scrib  else float("nan")

    print("\n=== Evaluation (Tiny U-Net) ===")
    print(f"Images evaluated: {len(rows)}")
    print(f"Mean IoU (BG):   {mean_iou_bg:.4f}")
    print(f"Mean IoU (FG):   {mean_iou_fg:.4f}")
    print(f"Mean mIoU:       {mean_miou:.4f}")
    print(f"Mean scrib-acc:  {mean_scrib:.4f}")

    # Optional CSV
    if args.save_csv:
        import csv
        with open(args.save_csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        print(f"[INFO] Saved per-image metrics to {args.save_csv}")

if __name__ == "__main__":
    main()
