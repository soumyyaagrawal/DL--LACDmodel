"""Image- and pixel-level metrics. All computed from real predictions + annotations."""
from __future__ import annotations

import numpy as np
from sklearn.metrics import (average_precision_score, confusion_matrix, f1_score, precision_score,
                             recall_score, roc_auc_score, roc_curve)


def image_metrics(labels, scores, threshold) -> dict:
    y, s = np.asarray(labels), np.asarray(scores)
    pred = (s >= threshold).astype(int)
    out = {"n": int(len(y)), "n_anomalous": int(y.sum()), "n_normal": int((1 - y).sum()), "threshold": float(threshold)}
    if y.min() != y.max():
        out["auroc"] = float(roc_auc_score(y, s))
        out["average_precision"] = float(average_precision_score(y, s))
    else:
        out["auroc"] = None  # undefined with a single class
    out["precision"] = float(precision_score(y, pred, zero_division=0))
    out["recall"] = float(recall_score(y, pred, zero_division=0))
    out["f1"] = float(f1_score(y, pred, zero_division=0))
    out["confusion_matrix"] = confusion_matrix(y, pred, labels=[0, 1]).tolist()  # [[TN,FP],[FN,TP]]
    return out


def roc_points(labels, scores):
    fpr, tpr, _ = roc_curve(labels, scores)
    return fpr.tolist(), tpr.tolist()


def pixel_metrics(maps: list[np.ndarray], masks: list[np.ndarray], pixel_thr: float, max_pixels: int = 3_000_000,
                  seed: int = 0) -> dict:
    """maps/masks at the SAME evaluation resolution (masks resized with nearest-neighbour).
    Pixel AUROC over all pixels (subsampled to max_pixels) of images that have masks; IoU/Dice at pixel_thr
    (selected on validation normals), averaged over anomalous images and also pooled."""
    if not maps:
        return {"n_images": 0}
    rng = np.random.default_rng(seed)
    flat_m = np.concatenate([m.ravel() for m in maps])
    flat_g = np.concatenate([g.ravel() for g in masks]).astype(np.uint8)
    out = {"n_images": len(maps), "pixel_threshold": float(pixel_thr)}
    if flat_g.min() != flat_g.max():
        idx = rng.choice(len(flat_g), min(max_pixels, len(flat_g)), replace=False)
        out["pixel_auroc"] = float(roc_auc_score(flat_g[idx], flat_m[idx]))
    else:
        out["pixel_auroc"] = None
    ious, dices = [], []
    ti = tu = tp = tg = 0
    for m, g in zip(maps, masks):
        p, g = m >= pixel_thr, g.astype(bool)
        inter, union = (p & g).sum(), (p | g).sum()
        if g.any():
            ious.append(inter / union if union else 0.0)
            dices.append(2 * inter / (p.sum() + g.sum()) if (p.sum() + g.sum()) else 0.0)
        ti, tu, tp, tg = ti + inter, tu + union, tp + p.sum(), tg + g.sum()
    out["iou_mean_anomalous"] = float(np.mean(ious)) if ious else None
    out["dice_mean_anomalous"] = float(np.mean(dices)) if dices else None
    out["iou_pooled"] = float(ti / tu) if tu else None
    out["dice_pooled"] = float(2 * ti / (tp + tg)) if (tp + tg) else None
    return out
