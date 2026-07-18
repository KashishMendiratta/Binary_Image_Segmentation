import os
import json
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from typing import Any, Tuple, Optional, List, Dict

# classical ML
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier

# ---------------------------------------
# Dataset I/O
# ---------------------------------------

def _open_image(path, convert_to):
    if convert_to == "RGB":
        return Image.open(path).convert("RGB")
    if convert_to == "grayscale":
        return Image.open(path).convert("L")
    return np.array(Image.open(path))

def _get_file_names(folder):
    return sorted([f for f in os.listdir(folder) if not f.startswith(".")])

def _load_images(folder_path, folder_name, convert_to):
    image_dir_path = os.path.join(folder_path, folder_name)
    filenames = _get_file_names(image_dir_path)
    filepaths = [os.path.join(image_dir_path, fn) for fn in filenames]
    return np.stack([_open_image(fp, convert_to) for fp in filepaths])

def _get_palette(folder_path, ground_truth_dir, filename):
    gt_dir_path = os.path.join(folder_path, ground_truth_dir)
    filepath = os.path.join(gt_dir_path, filename)
    return Image.open(filepath).getpalette()

def _get_filenames(folder_path, scribbles_dir):
    sc_dir_path = os.path.join(folder_path, scribbles_dir)
    return _get_file_names(sc_dir_path)

def load_dataset(
    folder_path: str,
    images_dir: str,
    scribbles_dir: str,
    ground_truth_dir: Optional[str] = None
):
    images = _load_images(folder_path, images_dir, "RGB")
    scribbles = _load_images(folder_path, scribbles_dir, "grayscale")
    filenames = _get_filenames(folder_path, scribbles_dir)
    if ground_truth_dir is None:
        return images, scribbles, filenames
    ground_truth = _load_images(folder_path, ground_truth_dir, None)
    palette = _get_palette(folder_path, ground_truth_dir, filenames[0])
    return images, scribbles, ground_truth, filenames, palette

def store_predictions(
    predictions: np.ndarray,
    folder_path: str,
    predictions_dir: str,
    filenames: List[str],
    palette: Any
):
    pred_dir_path = os.path.join(folder_path, predictions_dir)
    os.makedirs(pred_dir_path, exist_ok=True)
    for fn, pred in zip(filenames, predictions):
        assert pred.shape == (375, 500), f"Wrong size for {fn}: {pred.shape}"
        out = Image.fromarray(pred.astype(np.uint8), mode="P")
        out.putpalette(palette)
        out.save(os.path.join(pred_dir_path, fn))

# ---------------------------------------
# Viz
# ---------------------------------------

def _overlay_scribbles(img, scr, color_fg=(255,0,0), color_bg=(0,0,255), alpha=0.6):
    if img.ndim != 3 or img.shape[2] != 3:
        raise ValueError("Input image must be RGB")
    if scr.shape != img.shape[:2]:
        raise ValueError("Scribble must match image spatial size")
    out = img.copy().astype(np.float32)
    mfg = scr == 1
    mbg = scr == 0
    for m, col in [(mfg, color_fg), (mbg, color_bg)]:
        for c in range(3):
            out[..., c][m] = alpha * col[c] + (1 - alpha) * out[..., c][m]
    return out.astype(np.uint8)

def visualize(image, scribbles, ground_truth, prediction, alpha: float = 0.6):
    cmap = plt.get_cmap("bwr")
    fig, ax = plt.subplots(1, 3, figsize=(15, 5))
    ax[0].imshow(_overlay_scribbles(image, scribbles, alpha=alpha)); ax[0].set_title("Image + Scribbles")
    ax[1].imshow(ground_truth, cmap=cmap, vmin=0, vmax=1); ax[1].set_title("Ground Truth")
    ax[2].imshow(prediction,   cmap=cmap, vmin=0, vmax=1); ax[2].set_title("Model Prediction")
    for a in ax: a.axis("off")
    fig.tight_layout(); plt.show()

# ---------------------------------------
# Metrics
# ---------------------------------------

def _iou_binary_class(pred: np.ndarray, gt: np.ndarray, cls: int) -> float:
    inter = np.logical_and(pred == cls, gt == cls).sum()
    uni   = np.logical_or  (pred == cls, gt == cls).sum()
    return float(inter) / float(uni + 1e-6)

def compute_miou(pred: np.ndarray, gt: np.ndarray, ignore_val: int = 255) -> Dict[str, float]:
    if pred.shape != gt.shape:
        raise ValueError("pred and gt must have same spatial size")
    valid = (gt != ignore_val)
    if valid.sum() == 0:
        return {"iou_fg": np.nan, "iou_bg": np.nan, "miou": np.nan}
    p = pred[valid]; g = gt[valid]
    iou_fg = _iou_binary_class(p, g, 1)
    iou_bg = _iou_binary_class(p, g, 0)
    return {"iou_fg": iou_fg, "iou_bg": iou_bg, "miou": 0.5 * (iou_fg + iou_bg)}

# ---------------------------------------
# Small helpers
# ---------------------------------------

def post_process_binary(mask: np.ndarray, open_ks: int = 1, close_ks: int = 2, fill_holes: bool = False) -> np.ndarray:
    m = (mask > 0).astype(np.uint8)
    try:
        import cv2
        if open_ks > 0:
            k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (open_ks, open_ks))
            m = cv2.morphologyEx(m, cv2.MORPH_OPEN, k)
        if close_ks > 0:
            k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_ks, close_ks))
            m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, k)
        if fill_holes:
            h, w = m.shape
            ff = np.zeros((h+2, w+2), dtype=np.uint8)
            m_inv = 1 - m
            cv2.floodFill(m_inv.copy(), ff, (0,0), 255)
            holes = (m_inv == 0).astype(np.uint8)
            m = np.clip(m + holes, 0, 1)
    except Exception:
        pass
    return (m > 0).astype(np.uint8)

def dense_crf_refine(image_rgb: np.ndarray, prob_fg: np.ndarray, iters: int = 5) -> np.ndarray:
    """Optional; returns refined prob map in [0,1]."""
    try:
        import pydensecrf.densecrf as dcrf
        from pydensecrf.utils import unary_from_softmax, create_pairwise_gaussian, create_pairwise_bilateral
        H, W = prob_fg.shape
        probs = np.stack([1.0 - prob_fg, prob_fg], axis=0).clip(1e-6, 1 - 1e-6)
        U = unary_from_softmax(probs)
        d = dcrf.DenseCRF2D(W, H, 2)
        d.setUnaryEnergy(U)
        d.addPairwiseEnergy(create_pairwise_gaussian(sdims=(3, 3), shape=(H, W)), compat=3)
        d.addPairwiseEnergy(create_pairwise_bilateral(sdims=(50, 50), schan=(5, 5, 5), img=image_rgb, chdim=2), compat=5)
        Q = np.array(d.inference(iters)).reshape(2, H, W)
        return (Q[1] / (Q[0] + Q[1] + 1e-8)).astype(np.float32)
    except Exception:
        return prob_fg.astype(np.float32)

# ---------------------------------------
# Baseline KNN (RGB only)
# ---------------------------------------

def segment_with_knn(image: np.ndarray, scribble: np.ndarray, k: int = 3) -> np.ndarray:
    H, W, C = image.shape
    assert C == 3
    img_flat = image.reshape(-1, 3)
    sc_flat  = scribble.flatten()
    lab      = (sc_flat != 255)
    unlab    = ~lab
    Xtr      = img_flat[lab]
    ytr      = sc_flat[lab].astype(np.uint8)
    if Xtr.shape[0] == 0:
        return np.zeros((H, W), np.uint8)
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(Xtr, ytr)
    ypred = knn.predict(img_flat[unlab]) if unlab.any() else np.array([], np.uint8)
    out = np.zeros_like(sc_flat, np.uint8)
    out[lab]   = ytr
    out[unlab] = ypred
    return out.reshape(H, W)

# ---------------------------------------
# Feature builders for Pro models
# ---------------------------------------

def build_features(image: np.ndarray, xy_scale: float | None, add_lab: bool = True) -> np.ndarray:
    H, W, _ = image.shape
    imgf = (image.astype(np.float32) / 255.0) if image.dtype == np.uint8 else image.astype(np.float32)
    feats = [imgf.reshape(-1, 3)]
    if add_lab:
        try:
            import cv2
            lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB).astype(np.float32)
            L, A, B = lab[...,0]/100.0, (lab[...,1]-128.0)/128.0, (lab[...,2]-128.0)/128.0
            feats.append(np.stack([L,A,B], axis=2).reshape(-1,3))
        except Exception:
            try:
                from skimage.color import rgb2lab
                lab = rgb2lab(imgf).astype(np.float32)
                lab[...,0] /= 100.0; lab[...,1] /= 128.0; lab[...,2] /= 128.0
                feats.append(lab.reshape(-1,3))
            except Exception:
                pass
    if xy_scale is not None and xy_scale > 0:
        yy, xx = np.mgrid[0:H, 0:W]
        feats.append((xx.reshape(-1,1).astype(np.float32) * xy_scale))
        feats.append((yy.reshape(-1,1).astype(np.float32) * xy_scale))
    return np.concatenate(feats, axis=1)

def _compute_dist_features(scribble: np.ndarray, norm: float) -> Tuple[np.ndarray, np.ndarray]:
    H, W = scribble.shape
    try:
        from scipy.ndimage import distance_transform_edt as edt
        to_fg = edt((scribble != 1)) / (norm + 1e-6)
        to_bg = edt((scribble != 0)) / (norm + 1e-6)
    except Exception:
        to_fg = np.zeros((H, W), np.float32)
        to_bg = np.zeros((H, W), np.float32)
    return to_fg.reshape(-1,1).astype(np.float32), to_bg.reshape(-1,1).astype(np.float32)

def _apply_slic_smoothing(prob_map: np.ndarray, image_rgb: np.ndarray, n_segments: int = 400, compactness: float = 10.0) -> np.ndarray:
    H, W = prob_map.shape
    try:
        from skimage.segmentation import slic
        seg = slic(image_rgb, n_segments=n_segments, compactness=compactness, start_label=0)
        segf = seg.reshape(-1)
        pf   = prob_map.reshape(-1)
        sums = np.bincount(segf, weights=pf, minlength=segf.max()+1)
        cnts = np.bincount(segf, minlength=segf.max()+1) + 1e-6
        mean = sums / cnts
        return mean[segf].reshape(H, W).astype(np.float32)
    except Exception:
        return prob_map

def _keep_fg_components_touching_scribbles(mask_bin: np.ndarray, scribble: np.ndarray) -> np.ndarray:
    try:
        from scipy.ndimage import label
        lab, num = label((mask_bin > 0).astype(np.uint8), structure=np.ones((3,3), np.uint8))
        if num == 0: return mask_bin
        keep = np.zeros(num+1, dtype=bool)
        touched = lab[scribble == 1]
        keep[np.unique(touched)] = True
        keep[0] = False
        return np.where(keep[lab], 1, 0).astype(np.uint8)
    except Exception:
        return mask_bin

# ---------------------------------------
# Pro KNN
# ---------------------------------------

def segment_with_knn_pro(
    image: np.ndarray,
    scribble: np.ndarray,
    k: int = 7,
    add_lab: bool = True,
    add_xy: bool = True,
    add_dist: bool = True,
    xy_scale: Optional[float] = None,
    prob_thr: float = 0.5,
    bilateral_params: Optional[Tuple[int, float, float]] = (7, 0.1, 3.0),
    slic_params: Optional[Tuple[int, float]] = (400, 10.0),
    keep_fg_connected: bool = True,
    do_postprocess: bool = True,
    seed: int = 0,
) -> np.ndarray:
    H, W, _ = image.shape
    base = build_features(
        image,
        xy_scale=(1.0/max(H,W) if add_xy and xy_scale is None else (xy_scale if add_xy else 0.0)),
        add_lab=add_lab
    )
    feats_list = [base]
    if add_dist:
        dfg, dbg = _compute_dist_features(scribble, norm=float(max(H, W)))
        feats_list += [dfg, dbg]
    feats = np.concatenate(feats_list, axis=1).astype(np.float32)

    s = scribble.flatten()
    lab = (s != 255); unlab = ~lab
    Xtr = feats[lab]; ytr = s[lab].astype(np.uint8)
    if Xtr.shape[0] == 0:
        return np.zeros((H, W), np.uint8)
    uniq = np.unique(ytr)
    if uniq.size == 1:
        out = np.full(H*W, 0, np.uint8); out[lab] = ytr; out[unlab] = int(uniq[0])
        out = out.reshape(H, W)
        if keep_fg_connected: out = _keep_fg_components_touching_scribbles(out, scribble)
        return post_process_binary(out) if do_postprocess else out

    knn = KNeighborsClassifier(n_neighbors=k, weights="distance")
    knn.fit(Xtr, ytr)
    p_full = np.zeros(H*W, np.float32)
    if unlab.any():
        proba = knn.predict_proba(feats[unlab])
        if proba.shape[1] == 1:
            only = int(knn.classes_[0])
            p = np.ones(proba.shape[0], np.float32) if only == 1 else np.zeros(proba.shape[0], np.float32)
        else:
            idx = np.where(knn.classes_ == 1)[0]
            p = proba[:, idx[0]].astype(np.float32) if idx.size else np.zeros(proba.shape[0], np.float32)
        p_full[unlab] = p
    p_full[lab] = ytr.astype(np.float32)
    prob = p_full.reshape(H, W)

    if bilateral_params is not None:
        try:
            import cv2
            d, sc, ss = bilateral_params
            prob = cv2.bilateralFilter(prob.astype(np.float32), d=d, sigmaColor=sc, sigmaSpace=ss)
        except Exception:
            pass
    if slic_params is not None:
        nseg, comp = slic_params
        prob = _apply_slic_smoothing(prob, image, n_segments=nseg, compactness=comp)

    pred = (prob >= float(prob_thr)).astype(np.uint8)
    if keep_fg_connected:
        pred = _keep_fg_components_touching_scribbles(pred, scribble)
    if do_postprocess:
        pred = post_process_binary(pred, open_ks=1, close_ks=1, fill_holes=False)
    pred[scribble == 0] = 0; pred[scribble == 1] = 1
    return pred

# ---------------------------------------
# Tree PRO (RandomForest / ExtraTrees)
# ---------------------------------------

def segment_with_tree_pro(
    image: np.ndarray,
    scribble: np.ndarray,
    model: str = "randomforest",
    n_estimators: int = 300,
    max_depth: Optional[int] = None,
    add_lab: bool = True,
    add_xy: bool = False,
    add_dist: bool = True,
    xy_scale: float = 0.0,
    prob_thr: float = 0.5,
    bilateral_params: Optional[Tuple[int, float, float]] = (5, 0.05, 2.0),
    do_postprocess: bool = True,
    seed: int = 0,
) -> np.ndarray:
    H, W, _ = image.shape
    feats = build_features(image, xy_scale=(xy_scale if add_xy else 0.0), add_lab=add_lab)
    if add_dist:
        dfg, dbg = _compute_dist_features(scribble, norm=float(max(H, W)))
        feats = np.concatenate([feats, dfg, dbg], axis=1)

    s = scribble.flatten()
    lab = (s != 255); unlab = ~lab
    Xtr = feats[lab]; ytr = s[lab].astype(np.uint8)
    if Xtr.shape[0] == 0:
        return np.zeros((H, W), np.uint8)
    if np.unique(ytr).size == 1:
        out = np.full(H*W, 0, np.uint8); out[lab] = ytr; out[unlab] = int(ytr[0])
        out = out.reshape(H, W)
        if do_postprocess:
            out = post_process_binary(out, 1, 2, False)
        out[scribble == 0] = 0; out[scribble == 1] = 1
        return out

    Clf = ExtraTreesClassifier if model.lower().startswith("extra") else RandomForestClassifier
    clf = Clf(n_estimators=n_estimators, max_depth=max_depth, n_jobs=1, random_state=seed, class_weight="balanced")
    clf.fit(Xtr, ytr)

    p_full = np.zeros(H*W, np.float32)
    if unlab.any():
        proba = clf.predict_proba(feats[unlab])
        if proba.shape[1] == 1:
            only = int(clf.classes_[0])
            p = np.ones(proba.shape[0], np.float32) if only == 1 else np.zeros(proba.shape[0], np.float32)
        else:
            idx = np.where(clf.classes_ == 1)[0]
            p = proba[:, idx[0]].astype(np.float32) if idx.size else np.zeros(proba.shape[0], np.float32)
        p_full[unlab] = p
    p_full[lab] = ytr.astype(np.float32)
    prob = p_full.reshape(H, W)

    if bilateral_params is not None:
        try:
            import cv2
            d, sc, ss = bilateral_params
            prob = cv2.bilateralFilter(prob.astype(np.float32), d=d, sigmaColor=sc, sigmaSpace=ss)
        except Exception:
            pass

    pred = (prob >= float(prob_thr)).astype(np.uint8)
    if do_postprocess:
        pred = post_process_binary(pred, 1, 2, False)
    pred[scribble == 0] = 0; pred[scribble == 1] = 1
    return pred

# ---------------------------------------
# Graph-Cut (OpenCV GrabCut w/ scribble mask)
# ---------------------------------------

def segment_with_graphcut(image: np.ndarray, scribble: np.ndarray, gc_iters: int = 5) -> np.ndarray:
    """Maps scribbles to GrabCut labels and runs cv2.grabCut."""
    try:
        import cv2
    except Exception:
        # Fallback: return background
        return (scribble == 1).astype(np.uint8)

    H, W, _ = image.shape
    # GrabCut mask codes
    GC_BGD, GC_FGD, GC_PR_BGD, GC_PR_FGD = 0, 1, 2, 3
    mask = np.full((H, W), GC_PR_BGD, np.uint8)
    mask[scribble == 0] = GC_BGD
    mask[scribble == 1] = GC_FGD

    bgdModel = np.zeros((1, 65), np.float64)
    fgdModel = np.zeros((1, 65), np.float64)
    try:
        cv2.grabCut(image, mask, None, bgdModel, fgdModel, gc_iters, cv2.GC_INIT_WITH_MASK)
        out = np.where((mask == GC_FGD) | (mask == GC_PR_FGD), 1, 0).astype(np.uint8)
    except Exception:
        out = (scribble == 1).astype(np.uint8)

    out[scribble == 0] = 0
    out[scribble == 1] = 1
    return post_process_binary(out, 1, 2, False)

# ---------------------------------------
# Random-Walk (skimage)
# ---------------------------------------

def segment_with_randomwalk(image: np.ndarray, scribble: np.ndarray, beta: float = 90.0, mode: str = "cg") -> np.ndarray:
    try:
        from skimage.segmentation import random_walker
    except Exception:
        return (scribble == 1).astype(np.uint8)

    labels = np.full(scribble.shape, -1, np.int32)
    labels[scribble == 0] = 0
    labels[scribble == 1] = 1
    try:
        res = random_walker(image.astype(np.float32), labels, beta=beta, mode=mode)
        pred = (res == 1).astype(np.uint8)
    except Exception:
        pred = (scribble == 1).astype(np.uint8)

    pred[scribble == 0] = 0
    pred[scribble == 1] = 1
    return post_process_binary(pred, 1, 1, False)

# ---------------------------------------
# Tiny U-Net (scribble-supervised)
# ---------------------------------------

def _maybe_import_torch():
    try:
        import torch  # noqa
        import torch.nn as nn  # noqa
        import torch.nn.functional as F  # noqa
        from torch.utils.data import Dataset, DataLoader  # noqa
        return True
    except Exception:
        return False

def _to_tensor_img(img: np.ndarray, size: int):
    import torch
    import torch.nn.functional as F
    im = (img.astype(np.float32) / 255.0)
    im = np.transpose(im, (2,0,1))  # C,H,W
    t  = torch.from_numpy(im)[None]  # 1,C,H,W
    H, W = img.shape[:2]
    if size is not None and (H != size or W != size):
        t = F.interpolate(t, size=(size, size), mode="bilinear", align_corners=False)
    return t.squeeze(0)

class _ScribDataset:
    """Simple numpy-backed dataset for torch-free init; torch ops happen in loader."""
    def __init__(self, images: np.ndarray, scribbles: np.ndarray, resize: int = 320, hflip: bool = True):
        self.images = images
        self.scribs = scribbles
        self.resize = resize
        self.hflip  = hflip

    def __len__(self): return len(self.images)

    def __getitem__(self, i):
        import torch
        import torch.nn.functional as F
        img = self.images[i]; scr = self.scribs[i]
        if self.hflip and np.random.rand() < 0.5:
            img = img[:, ::-1, :].copy(); scr = scr[:, ::-1].copy()
        x = _to_tensor_img(img, self.resize)           # C,Hs,Ws
        y = torch.from_numpy(scr.astype(np.int64))[None]  # 1,H,W (original)
        y = y.float()  # keep as float; we'll resize and build mask
        y = F.interpolate(y[None], size=(self.resize, self.resize), mode="nearest").squeeze(0)  # 1,Hs,Ws
        y = y.squeeze(0)  # Hs,Ws (values 0,1,255)
        return x, y

class TinyUNet:
    """Tiny UNet wrapper so util.py avoids importing torch at module import time."""
    def __init__(self, in_ch=3, base=16, out_ch=1):
        import torch.nn as nn
        class Block(nn.Module):
            def __init__(self, c1, c2):
                super().__init__()
                self.conv = nn.Sequential(
                    nn.Conv2d(c1, c2, 3, padding=1), nn.ReLU(inplace=True),
                    nn.Conv2d(c2, c2, 3, padding=1), nn.ReLU(inplace=True),
                )
            def forward(self, x): return self.conv(x)

        class Net(nn.Module):
            def __init__(self, in_ch, base, out_ch):
                super().__init__()
                self.b1 = Block(in_ch, base)
                self.p1 = nn.MaxPool2d(2)
                self.b2 = Block(base, base*2)
                self.p2 = nn.MaxPool2d(2)
                self.b3 = Block(base*2, base*4)
                self.u2 = nn.ConvTranspose2d(base*4, base*2, 2, stride=2)
                self.b4 = Block(base*4, base*2)
                self.u1 = nn.ConvTranspose2d(base*2, base, 2, stride=2)
                self.b5 = Block(base*2, base)
                self.out = nn.Conv2d(base, out_ch, 1)
            def forward(self, x):
                x1 = self.b1(x); x2 = self.b2(self.p1(x1)); x3 = self.b3(self.p2(x2))
                x  = self.u2(x3); x = self.b4(np_concat(x, x2))
                x  = self.u1(x);  x = self.b5(np_concat(x, x1))
                return self.out(x)

        # tiny helper for channel concat without importing torch here
        global np_concat
        def np_concat(a, b):
            import torch
            return torch.cat([a, b], dim=1)

        self.Net = Net(in_ch, base, out_ch)

def train_unet_scribbles(
    images: np.ndarray,
    scribbles: np.ndarray,
    resize: int = 320,
    epochs: int = 12,
    batch_size: int = 2,
    lr: float = 1e-3,
    seed: int = 0,
    save_path: str = "tiny_unet.pt",
) -> Optional[str]:
    """Trains a tiny UNet with masked BCE loss (only scribbled pixels contribute)."""
    if not _maybe_import_torch():
        print("[UNet] PyTorch not found; skipping UNet path.")
        return None

    import torch, random
    from torch.utils.data import DataLoader
    import torch.nn as nn
    import torch.nn.functional as F
    torch.manual_seed(seed); np.random.seed(seed); random.seed(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = TinyUNet().Net.to(device)

    ds  = _ScribDataset(images, scribbles, resize=resize, hflip=True)
    dl  = DataLoader(ds, batch_size=batch_size, shuffle=True, num_workers=0)

    opt = torch.optim.Adam(net.parameters(), lr=lr)
    net.train()
    eps = 1e-6

    from tqdm.auto import tqdm
    for ep in range(1, epochs+1):
        running = 0.0; denom = 0.0
        for x, y in tqdm(dl, desc=f"[UNet] epoch {ep}/{epochs}", unit="batch", leave=False):
            x = x.to(device)            # B,C,H,W in [0,1]
            y = y.to(device)            # B,H,W with 0,1,255
            mask = (y != 255).float()   # B,H,W
            tgt  = (y == 1).float()     # B,H,W

            logits = net(x).squeeze(1)  # B,H,W
            loss_map = torch.nn.functional.binary_cross_entropy_with_logits(logits, tgt, reduction="none")
            loss = (loss_map * mask).sum() / (mask.sum() + eps)

            opt.zero_grad(); loss.backward(); opt.step()
            running += loss.item(); denom += 1.0
        print(f"[UNet] epoch {ep}: loss={running / max(denom,1):.4f}")

    torch.save(net.state_dict(), save_path)
    print(f"[UNet] saved to {save_path}")
    return save_path

def unet_predict_proba(
    image: np.ndarray,
    scribble: np.ndarray,
    model_path: str,
    resize: int = 320,
    tta: str = "hflip",
    crf_iters: int = 0,
) -> np.ndarray:
    """Returns prob_fg in [0,1] at original size."""
    if not _maybe_import_torch():
        # no torch: return zeros prob
        return np.zeros(image.shape[:2], np.float32)

    import torch
    import torch.nn.functional as F
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = TinyUNet().Net.to(device)
    net.load_state_dict(torch.load(model_path, map_location=device))
    net.eval()

    def _forward(img):
        x = _to_tensor_img(img, resize).unsqueeze(0).to(device)  # 1,C,Rs,Rs
        with torch.no_grad():
            logit = net(x).squeeze(0).squeeze(0)  # Rs,Rs
            prob  = torch.sigmoid(logit)[None,None]
            prob  = F.interpolate(prob, size=img.shape[:2], mode="bilinear", align_corners=False).squeeze().cpu().numpy().astype(np.float32)
        return prob

    if tta == "none":
        prob = _forward(image)
    elif tta == "hflip":
        p1 = _forward(image)
        p2 = _forward(image[:, ::-1, :])[:, ::-1]
        prob = 0.5*(p1+p2)
    else:  # 4way
        p = []
        p.append(_forward(image))
        p.append(_forward(image[:, ::-1, :])[:, ::-1])
        p.append(_forward(image[::-1, :, :])[::-1, :])
        p.append(_forward(image[::-1, ::-1, :])[::-1, ::-1])
        prob = np.mean(p, axis=0)

    if crf_iters > 0 and image.dtype == np.uint8:
        prob = dense_crf_refine(image, prob.astype(np.float32), iters=crf_iters)

    return np.clip(prob, 0.0, 1.0).astype(np.float32)

def threshold_with_scribbles(prob: np.ndarray, scribble: np.ndarray, grid: np.ndarray = np.linspace(0.35, 0.65, 7)) -> float:
    """Pick threshold maximizing IoU on labeled pixels only."""
    L = (scribble != 255)
    if L.sum() == 0:
        return 0.5
    best_t, best_iou = 0.5, -1.0
    lab = scribble[L]
    for t in grid:
        pred = (prob[L] >= t).astype(np.uint8)
        inter = np.logical_and(pred == 1, lab == 1).sum()
        uni   = np.logical_or  (pred == 1, lab == 1).sum()
        iou   = float(inter) / float(uni + 1e-6)
        if iou > best_iou:
            best_iou, best_t = iou, float(t)
    return best_t
