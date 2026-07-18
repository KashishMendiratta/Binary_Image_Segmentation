# challenge.py — KNN / Tree / GraphCut / RandomWalk + Tiny U-Net + CV tuner + ensemble + fast refinement (robust no-tune)

import argparse, os, json
import numpy as np
from tqdm.auto import tqdm

from util import (
    load_dataset, store_predictions, visualize,
    # classical
    segment_with_knn, segment_with_knn_pro, segment_with_tree_pro,
    segment_with_graphcut, segment_with_randomwalk,
    # unet
    train_unet_scribbles, unet_predict_proba, threshold_with_scribbles,
    # misc
    compute_miou, post_process_binary
)

# ----------------------------
# Metrics over a batch
# ----------------------------
def batch_ious(preds, gts):
    fg, bg, miou = [], [], []
    for p, g in zip(preds, gts):
        r = compute_miou(p, g)
        fg.append(r["iou_fg"]); bg.append(r["iou_bg"]); miou.append(r["miou"])
    return {"iou_fg": float(np.nanmean(fg)),
            "iou_bg": float(np.nanmean(bg)),
            "miou":   float(np.nanmean(miou))}

# ----------------------------
# Family-aware predictor + TTA
# ----------------------------
def _predict_once(img, scr, cfg):
    fam = cfg["family"]
    if fam == "knn":
        return segment_with_knn_pro(
            img, scr,
            k=cfg.get("k", 7),
            add_lab=True, add_xy=False, add_dist=cfg.get("add_dist", True),
            xy_scale=0.0, prob_thr=cfg.get("prob_thr", 0.5),
            bilateral_params=cfg.get("bilateral_params", (7,0.1,3.0)),
            slic_params=None, keep_fg_connected=True, do_postprocess=True, seed=0
        )
    if fam == "tree":
        return segment_with_tree_pro(
            img, scr,
            model=cfg.get("tree_model", "randomforest"),
            n_estimators=cfg.get("n_estimators", 300),
            add_lab=True, add_xy=False, add_dist=cfg.get("add_dist", True),
            xy_scale=0.0, prob_thr=cfg.get("prob_thr", 0.5),
            bilateral_params=cfg.get("bilateral_params", (5,0.05,2.0)),
            do_postprocess=True, seed=0
        )
    if fam == "gc":
        return segment_with_graphcut(img, scr, gc_iters=int(cfg.get("gc_iters", 5)))
    if fam == "rw":
        return segment_with_randomwalk(img, scr, beta=float(cfg.get("beta", 90.0)), mode=str(cfg.get("rw_mode", "cg")))
    raise ValueError(f"Unknown family: {fam}")

def _predict_tta(img, scr, cfg):
    tta = cfg.get("tta", "hflip")
    if tta == "none":
        p = _predict_once(img, scr, cfg)

    elif tta == "hflip":
        p1 = _predict_once(img, scr, cfg)
        p2 = _predict_once(np.ascontiguousarray(img[:, ::-1, :]),
                           np.ascontiguousarray(scr[:, ::-1]), cfg)[:, ::-1]
        p  = ((p1.astype(np.uint8) + p2.astype(np.uint8)) >= 1).astype(np.uint8)

    elif tta == "4way":
        # 0: as-is, 1: hflip, 2: vflip, 3: hvflip
        p0 = _predict_once(img, scr, cfg)
        p1 = _predict_once(np.ascontiguousarray(img[:, ::-1, :]),
                           np.ascontiguousarray(scr[:, ::-1]), cfg)[:, ::-1]
        p2 = _predict_once(np.ascontiguousarray(img[::-1, :, :]),
                           np.ascontiguousarray(scr[::-1, :]), cfg)[::-1, :]
        p3 = _predict_once(np.ascontiguousarray(img[::-1, ::-1, :]),
                           np.ascontiguousarray(scr[::-1, ::-1]), cfg)[::-1, ::-1]
        acc = p0.astype(np.uint8) + p1.astype(np.uint8) + p2.astype(np.uint8) + p3.astype(np.uint8)
        p = (acc >= 2).astype(np.uint8)

    else:
        p = _predict_once(img, scr, cfg)

    # Always enforce scribbles
    p[scr == 0] = 0
    p[scr == 1] = 1
    return p

# ----------------------------
# Config sampler + CV
# ----------------------------
def _pick(rng, opts): return opts[int(rng.integers(0, len(opts)))]

def sample_cfg(rng):
    r = rng.random()
    if r < 0.35:
        return {"family":"gc", "gc_iters": int(_pick(rng,[3,5,8])), "prob_thr":0.55, "tta": _pick(rng,["none","hflip"])}
    if r < 0.65:
        return {
            "family":"tree", "tree_model": _pick(rng,["randomforest","extratrees"]),
            "n_estimators": int(_pick(rng,[150,250,350,450])),
            "add_dist": bool(rng.integers(0,2)), "prob_thr": _pick(rng,[0.50,0.55]),
            "bilateral_params": _pick(rng,[None,(5,0.05,2.0)]), "tta": _pick(rng,["none","hflip"])
        }
    else:
        return {
            "family":"knn", "k": int(_pick(rng,[3,5,7,9])),
            "add_dist": bool(rng.integers(0,2)), "prob_thr": _pick(rng,[0.50,0.55,0.60]),
            "bilateral_params": _pick(rng,[None,(5,0.05,2.0),(7,0.08,3.0)]), "tta": _pick(rng,["none","hflip"])
        }

def evaluate_cfg_cv(cfg, images, scribbles, gts, n_splits=3, seed=0):
    rng = np.random.default_rng(seed)
    N = len(images); idx = np.arange(N); rng.shuffle(idx); folds = np.array_split(idx, n_splits)
    fold_scores = []
    for fi, val_idx in enumerate(folds, start=1):
        miou_list, fg_list = [], []
        for i in tqdm(val_idx, desc=f"  fold {fi}/{n_splits} [{cfg['family']}]", unit="img", leave=False):
            pred = _predict_tta(images[i], scribbles[i], cfg)
            res  = compute_miou(pred, gts[i])
            miou_list.append(res["miou"]); fg_list.append(res["iou_fg"])
        fold_scores.append(0.8 * float(np.nanmean(miou_list)) + 0.2 * float(np.nanmean(fg_list)))
    return float(np.mean(fold_scores))

def tune_and_select(images, scribbles, gts, n_trials=30, n_splits=3, seed=0, log_path="tuning_log.jsonl"):
    rng = np.random.default_rng(seed)
    best_cfg, best_score = None, -1.0
    tried = set()
    with open(log_path, "a", buffering=1) as lf:
        for t in range(1, n_trials+1):
            cfg = sample_cfg(rng)
            key = tuple(sorted((k, str(v)) for k, v in cfg.items()))
            if key in tried: continue
            tried.add(key)
            print(f"[tune] trial {t:02d}/{n_trials}: cfg={cfg}")
            score = evaluate_cfg_cv(cfg, images, scribbles, gts, n_splits=n_splits, seed=seed)
            print(f"[tune] → cv_score={score:.4f}")
            lf.write(json.dumps({"trial": t, "cfg": cfg, "cv_score": score}) + "\n")
            if score > best_score:
                best_cfg, best_score = cfg, score
                print(f"[tune] ★ new best: {best_score:.4f} with {best_cfg}")
    print(f"[tune] BEST cv_score={best_score:.4f} with cfg={best_cfg}")
    return best_cfg

def top_configs_from_log(log_path="tuning_log.jsonl", topk=2):
    if not os.path.exists(log_path): return []
    rows = []
    with open(log_path, "r") as f:
        for line in f:
            try:
                rec = json.loads(line.strip())
                rows.append((rec["cv_score"], rec["cfg"]))
            except Exception:
                pass
    rows.sort(reverse=True, key=lambda x: x[0])
    uniq = []
    for _, cfg in rows:
        if cfg not in uniq:
            uniq.append(cfg)
        if len(uniq) >= topk: break
    return uniq

# ----- Fast post-hoc refinement helpers (GrabCut + connectivity + morphology + optional CRF)

def _keep_touching_fg(mask: np.ndarray, scribble: np.ndarray) -> np.ndarray:
    try:
        from scipy.ndimage import label
        lab, num = label((mask == 1).astype(np.uint8), structure=np.ones((3,3), np.uint8))
        if num == 0:
            return mask
        keep = np.zeros(num + 1, dtype=bool)
        touched = lab[scribble == 1]
        keep[np.unique(touched)] = True
        keep[0] = False
        return np.where(keep[lab], 1, 0).astype(np.uint8)
    except Exception:
        return mask

def _morph_clean(mask: np.ndarray, open_ks=1, close_ks=2) -> np.ndarray:
    try:
        import cv2
        m = mask.astype(np.uint8)
        if open_ks > 0:
            k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (open_ks, open_ks))
            m = cv2.morphologyEx(m, cv2.MORPH_OPEN, k)
        if close_ks > 0:
            k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_ks, close_ks))
            m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, k)
        return (m > 0).astype(np.uint8)
    except Exception:
        return (mask > 0).astype(np.uint8)

def _grabcut_refine(img_rgb: np.ndarray, scrib: np.ndarray, init_mask: np.ndarray, iters: int = 3) -> np.ndarray:
    import cv2
    H, W = init_mask.shape
    gc_mask = np.full((H, W), cv2.GC_PR_BGD, dtype=np.uint8)
    gc_mask[scrib == 0] = cv2.GC_BGD
    gc_mask[scrib == 1] = cv2.GC_FGD
    gc_mask[(scrib != 0) & (scrib != 1)] = cv2.GC_PR_BGD
    gc_mask[(init_mask == 1) & (scrib != 0)] = cv2.GC_PR_FGD
    bgdModel = np.zeros((1,65), np.float64)
    fgdModel = np.zeros((1,65), np.float64)
    cv2.grabCut(img_rgb, gc_mask, None, bgdModel, fgdModel, iters, cv2.GC_INIT_WITH_MASK)
    out = np.where((gc_mask == cv2.GC_FGD) | (gc_mask == cv2.GC_PR_FGD), 1, 0).astype(np.uint8)
    out[scrib == 0] = 0; out[scrib == 1] = 1
    return out

def _maybe_crf(img_rgb: np.ndarray, prob: np.ndarray, iters: int = 3) -> np.ndarray:
    try:
        import pydensecrf.densecrf as dcrf
        from pydensecrf.utils import unary_from_softmax, create_pairwise_gaussian, create_pairwise_bilateral
        H, W = prob.shape
        probs = np.stack([1.0 - prob, prob], axis=0).clip(1e-6, 1-1e-6)
        U = unary_from_softmax(probs)
        d = dcrf.DenseCRF2D(W, H, 2)
        d.setUnaryEnergy(U)
        d.addPairwiseEnergy(create_pairwise_gaussian(sdims=(3,3), shape=(H,W)), compat=3)
        d.addPairwiseEnergy(create_pairwise_bilateral(sdims=(50,50), schan=(5,5,5), img=img_rgb, chdim=2), compat=5)
        Q = np.array(d.inference(iters)).reshape(2, H, W)
        return (Q[1] / (Q[0] + Q[1] + 1e-8)).astype(np.float32)
    except Exception:
        return prob

def refine_predictions_folder(root: str, pred_dir: str, out_dir: str,
                              grabcut_iters: int = 3, crf_iters: int = 3) -> None:
    from PIL import Image
    imgs, scrs, fns = load_dataset(root, "images", "scribbles")
    os.makedirs(os.path.join(root, out_dir), exist_ok=True)
    first = Image.open(os.path.join(root, pred_dir, fns[0]))
    palette = first.getpalette()
    for i, name in enumerate(tqdm(fns, desc=f"Refine {os.path.basename(root)}", unit="img")):
        img = imgs[i]; scr = scrs[i]
        init = np.array(Image.open(os.path.join(root, pred_dir, name))).astype(np.uint8)
        m = _grabcut_refine(img, scr, init, iters=grabcut_iters)
        m = _keep_touching_fg(m, scr)
        m = _morph_clean(m, open_ks=1, close_ks=2)
        if crf_iters > 0:
            prob = _maybe_crf(img, m.astype(np.float32), iters=crf_iters)
            m = (prob >= 0.5).astype(np.uint8)
        m[scr == 0] = 0; m[scr == 1] = 1
        assert m.shape == (375, 500), f"Refined wrong size: {m.shape}"
        out = Image.fromarray(m, mode="P"); out.putpalette(palette)
        out.save(os.path.join(root, out_dir, name))

# ----------------------------
# Main
# ----------------------------
def main():
    ap = argparse.ArgumentParser("Segmentation Challenge — classical + tiny UNet + tuner/ensemble")
    ap.add_argument("--data_root", default="dataset")
    ap.add_argument("--train_split", default="train")
    ap.add_argument("--test_split", default="test1")
    ap.add_argument("--trials", type=int, default=50)
    ap.add_argument("--folds",  type=int, default=3)
    ap.add_argument("--final_tta", choices=["none","hflip","4way"], default="4way")
    ap.add_argument("--ensemble", choices=["none","avg_top2","avg_all"], default="avg_top2")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--visualize", action="store_true")

    # UNet options
    ap.add_argument("--use_unet", action="store_true")
    ap.add_argument("--unet_epochs", type=int, default=12)
    ap.add_argument("--unet_bs",     type=int, default=2)
    ap.add_argument("--unet_resize", type=int, default=320)
    ap.add_argument("--unet_lr",     type=float, default=1e-3)
    ap.add_argument("--unet_tta",    choices=["none","hflip","4way"], default="hflip")
    ap.add_argument("--unet_crf_iters", type=int, default=0)
    ap.add_argument("--unet_model_path", default="tiny_unet.pt")

    # Fast refinement options
    ap.add_argument("--refine_only", action="store_true")
    ap.add_argument("--refine_root", default="dataset/test1")
    ap.add_argument("--refine_pred_dir", default="predictions")
    ap.add_argument("--refine_out_dir",  default="predictions_refined")
    ap.add_argument("--refine_after_inference", action="store_true")
    ap.add_argument("--grabcut_iters", type=int, default=3)
    ap.add_argument("--crf_iters", type=int, default=3)

    args = ap.parse_args()

    # --- fast refine-only mode ---
    if args.refine_only:
        refine_predictions_folder(
            root=args.refine_root,
            pred_dir=args.refine_pred_dir,
            out_dir=args.refine_out_dir,
            grabcut_iters=args.grabcut_iters,
            crf_iters=args.crf_iters,
        )
        print("\nRefine-only finished.")
        return

    # ---- Load data
    train_root = os.path.join(args.data_root, args.train_split)
    images_train, scrib_train, gt_train, fnames_train, palette = load_dataset(
        train_root, "images", "scribbles", "ground_truth"
    )

    # ---- Baseline KNN for reference
    pred_train_base = []
    for img, scr in tqdm(zip(images_train, scrib_train), total=len(images_train),
                         desc="Baseline KNN (train)", unit="img"):
        pred_train_base.append(segment_with_knn(img, scr, k=3))
    pred_train_base = np.stack(pred_train_base, axis=0)
    base_scores = batch_ious(pred_train_base, gt_train)
    print("[Baseline KNN(k=3)] mIoU={:.4f} BG={:.4f} FG={:.4f}".format(
        base_scores["miou"], base_scores["iou_bg"], base_scores["iou_fg"]))

    # ---- Tune classical families (or load defaults when --trials 0)
    best_cfg = None
    if args.trials > 0:
        best_cfg = tune_and_select(
            images_train, scrib_train, gt_train,
            n_trials=args.trials, n_splits=args.folds, seed=args.seed,
            log_path="tuning_log.jsonl"
        )
        if best_cfg is not None:
            with open("best_config.json","w") as f:
                json.dump(best_cfg, f, indent=2)
            print("Saved best classical config to best_config.json")
    else:
        # 1) try to reuse last best_config.json
        if os.path.exists("best_config.json"):
            try:
                with open("best_config.json","r") as f:
                    best_cfg = json.load(f)
                print("[no-tune] loaded best_config.json:", best_cfg)
            except Exception:
                best_cfg = None
        # 2) else try top from log
        if best_cfg is None:
            tops = top_configs_from_log("tuning_log.jsonl", topk=1)
            if tops:
                best_cfg = tops[0]
                print("[no-tune] loaded top from tuning_log.jsonl:", best_cfg)
        # 3) else fall back to a strong default
        if best_cfg is None:
            best_cfg = {"family":"gc", "gc_iters":3, "prob_thr":0.55, "tta":"hflip"}
            print("[no-tune] using default:", best_cfg)

    # prepare list of configs to ensemble later (guaranteed non-empty)
    cfgs_for_ensemble = []
    if best_cfg is not None:
        cfgs_for_ensemble.append(best_cfg)
    if args.ensemble in ("avg_top2","avg_all"):
        topk = 4 if args.ensemble == "avg_all" else 2
        extra = top_configs_from_log("tuning_log.jsonl", topk=topk)
        for c in extra:
            if c not in cfgs_for_ensemble:
                cfgs_for_ensemble.append(c)
    if not cfgs_for_ensemble:
        cfgs_for_ensemble = [{"family":"gc", "gc_iters":3, "prob_thr":0.55, "tta":"hflip"}]

    # ---- Optional: train Tiny UNet (scribble-supervised)
    unet_path = None
    if args.use_unet:
        unet_path = train_unet_scribbles(
            images_train, scrib_train,
            resize=args.unet_resize, epochs=args.unet_epochs,
            batch_size=args.unet_bs, lr=args.unet_lr,
            seed=args.seed, save_path=args.unet_model_path
        )

    # ---- Build train predictions (and probs for UNet)
    train_layers = []   # list of float probs to ensemble
    for cfg in cfgs_for_ensemble:
        preds = []
        # honor requested final_tta (now supports 4way)
        cfg_final = {**cfg, "tta": args.final_tta}
        for img, scr in tqdm(zip(images_train, scrib_train), total=len(images_train),
                             desc=f"Ensemble(train) {cfg['family']}", unit="img"):
            preds.append(_predict_tta(img, scr, cfg_final))
        preds = np.stack(preds, axis=0)
        train_layers.append(preds.astype(np.float32))  # 0/1 float
    if unet_path is not None:
        prob_list = []
        for img, scr in tqdm(zip(images_train, scrib_train), total=len(images_train),
                             desc="UNet(train) prob", unit="img"):
            prob = unet_predict_proba(
                img, scr, model_path=unet_path,
                resize=args.unet_resize, tta=args.unet_tta,
                crf_iters=args.unet_crf_iters
            )
            prob_list.append(prob)
        train_layers.append(np.stack(prob_list, axis=0))  # (N,H,W) in [0,1]

    # ---- Ensemble train
    if args.ensemble == "none" and unet_path is None:
        pred_train_best = (train_layers[0] >= 0.5).astype(np.uint8)
    else:
        probs = np.mean(train_layers, axis=0)  # (N,H,W)
        out = []
        for pr, sc in zip(probs, scrib_train):
            thr = threshold_with_scribbles(pr, sc)
            m = (pr >= thr).astype(np.uint8)
            m[sc == 0] = 0; m[sc == 1] = 1
            out.append(post_process_binary(m, 1, 2, False))
        pred_train_best = np.stack(out, axis=0)

    # ---- Report train mIoU
    best_scores = batch_ious(pred_train_best, gt_train)
    print("[Tuned+Ensemble (train)] mIoU={:.4f} BG={:.4f} FG={:.4f}".format(
        best_scores["miou"], best_scores["iou_bg"], best_scores["iou_fg"]))

    # ---- Save train preds
    store_predictions(pred_train_best, train_root, "predictions", fnames_train, palette)

    if args.visualize:
        i = np.random.randint(len(images_train))
        visualize(images_train[i], scrib_train[i], gt_train[i], pred_train_best[i])

    # ---- Test split
    test_root = os.path.join(args.data_root, args.test_split)
    images_test, scrib_test, fnames_test = load_dataset(test_root, "images", "scribbles")

    test_layers = []
    for cfg in cfgs_for_ensemble:
        preds = []
        cfg_final = {**cfg, "tta": args.final_tta}
        for img, scr in tqdm(zip(images_test, scrib_test), total=len(images_test),
                             desc=f"Ensemble(test) {cfg['family']}", unit="img"):
            preds.append(_predict_tta(img, scr, cfg_final))
        test_layers.append(np.stack(preds, axis=0).astype(np.float32))
    if unet_path is not None:
        prob_list = []
        for img, scr in tqdm(zip(images_test, scrib_test), total=len(images_test),
                             desc="UNet(test) prob", unit="img"):
            prob = unet_predict_proba(
                img, scr, model_path=unet_path,
                resize=args.unet_resize, tta=args.unet_tta,
                crf_iters=args.unet_crf_iters
            )
            prob_list.append(prob)
        test_layers.append(np.stack(prob_list, axis=0))

    if args.ensemble == "none" and unet_path is None:
        pred_test = (test_layers[0] >= 0.5).astype(np.uint8)
    else:
        probs = np.mean(test_layers, axis=0)
        out = []
        for pr, sc in zip(probs, scrib_test):
            thr = threshold_with_scribbles(pr, sc)
            m = (pr >= thr).astype(np.uint8)
            m[sc == 0] = 0; m[sc == 1] = 1
            out.append(post_process_binary(m, 1, 2, False))
        pred_test = np.stack(out, axis=0)

    store_predictions(pred_test, test_root, "predictions", fnames_test, palette)
    print("\nDone. Train + test predictions saved.")

    if args.refine_after_inference:
        refine_predictions_folder(
            root=test_root,
            pred_dir="predictions",
            out_dir="predictions_refined",
            grabcut_iters=args.grabcut_iters,
            crf_iters=args.crf_iters,
        )

if __name__ == "__main__":
    main()
