"""Evaluate detectors on the SAME eligible test samples; per category, per anomaly type."""
from __future__ import annotations

import time

import numpy as np
import torch
from PIL import Image

from lacd.datasets.loco import Sample, load_mask, load_rgb, preprocess, select
from lacd.evaluation.metrics import image_metrics, pixel_metrics, roc_points
from lacd.models.detector import Detector


def score_samples(det: Detector, test: list[Sample], keep_maps: bool = True) -> dict:
    S = det.cfg.data.image_size
    rows, maps, masks, preds = [], [], [], []
    for s in test:
        img = load_rgb(s.path)
        x = preprocess(img, S)
        fin = det.finalize(det.raw(x))
        rows.append({"path": s.path, "label": s.label, "type": s.anomaly_type, "subtype": s.subtype,
                     "score": float(fin["score"]), "app_z": float(fin["app_z"]), "rel_z": float(fin["rel_z"]),
                     "n_violations": len(fin["violations"]),
                     "violation_types": sorted({v["type"] for v in fin["violations"]})})
        m = load_mask(s, (S, S))
        if keep_maps and m is not None:
            maps.append(fin["map"]); masks.append(m)
        preds.append(fin)
    return {"rows": rows, "maps": maps, "masks": masks, "preds": preds}


def evaluate_detector(det: Detector, samples: list[Sample]) -> dict:
    test = select(samples, "test")
    if not test:
        raise RuntimeError("No test samples found.")
    res = score_samples(det, test)
    rows = res["rows"]
    y = np.array([r["label"] for r in rows])
    s = np.array([r["score"] for r in rows])
    thr, pthr = det.cal["threshold"], det.cal["pixel_threshold"]
    out = {"model": det.kind, "checkpoint_id": det.checkpoint_id, "threshold": thr,
           "threshold_procedure": det.cal.get("threshold_procedure"), "pixel_threshold": pthr,
           "pixel_threshold_procedure": det.cal.get("pixel_threshold_procedure"),
           "weights_source": det.extractor.weights_source, "overall": image_metrics(y, s, thr)}
    fpr, tpr = roc_points(y, s) if y.min() != y.max() else ([], [])
    out["roc"] = {"fpr": fpr, "tpr": tpr}
    types = sorted({r["type"] for r in rows if r["label"] == 1})
    out["by_type"] = {}
    for t in types:  # anomalies of type t vs ALL normal test images
        idx = [i for i, r in enumerate(rows) if r["label"] == 0 or r["type"] == t]
        out["by_type"][t] = image_metrics(y[idx], s[idx], thr)
    subs = sorted({r["subtype"] for r in rows if r["subtype"]})
    out["by_subtype"] = {}
    for t in subs:
        idx = [i for i, r in enumerate(rows) if r["label"] == 0 or r["subtype"] == t]
        out["by_subtype"][t] = image_metrics(y[idx], s[idx], thr)
    # ablation: appearance-only and relation-only components of the same model
    out["components"] = {"appearance_z": image_metrics(y, [r["app_z"] for r in rows], thr)}
    if det.kind == "relation":
        out["components"]["relation_z"] = image_metrics(y, [r["rel_z"] for r in rows], thr)
    out["normal_false_positive_rate"] = float(((s >= thr) & (y == 0)).sum() / max(1, (y == 0).sum()))
    out["pixel"] = {"overall": pixel_metrics(res["maps"], res["masks"], pthr)}
    for t in types:
        keep = [i for i, r in enumerate(rows) if (r["label"] == 0 or r["type"] == t) and
                load_mask(test[i], (1, 1)) is not None]
        mm = {"maps": [], "masks": []}
        for i in keep:
            mk = load_mask(test[i], (det.cfg.data.image_size,) * 2)
            mm["maps"].append(res["preds"][i]["map"]); mm["masks"].append(mk)
        out["pixel"][t] = pixel_metrics(mm["maps"], mm["masks"], pthr)
    out["missing_masks"] = int(sum(1 for r in rows if r["label"] == 1) - sum(1 for m in res["masks"] if m.any()))
    out["rows"] = rows
    out["latency"] = benchmark_latency(det, [s_.path for s_ in test][:20])
    out["_internal"] = res  # stripped before JSON
    return out


def benchmark_latency(det: Detector, paths: list[str], warmup: int = 3) -> dict:
    imgs = [load_rgb(p) for p in paths]
    if not imgs:
        return {}
    dev = det.device
    if dev.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    for im in imgs[:warmup]:
        det.predict(im)
    ts = []
    for im in imgs:
        if dev.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        det.predict(im)
        if dev.type == "cuda":
            torch.cuda.synchronize()
        ts.append((time.perf_counter() - t0) * 1000)
    return {"device": str(dev), "n_images": len(imgs), "image_size_input": det.cfg.data.image_size,
            "mean_ms": float(np.mean(ts)), "median_ms": float(np.median(ts)), "std_ms": float(np.std(ts)),
            "methodology": f"{warmup} warm-up runs; end-to-end predict() incl. preprocessing and map resize; "
                           + ("CUDA synchronised" if dev.type == "cuda" else "CPU wall-clock"),
            "peak_gpu_mem_mb": float(torch.cuda.max_memory_allocated() / 2**20) if dev.type == "cuda" else None,
            "peak_gpu_mem_note": None if dev.type == "cuda" else "unavailable (CPU run)"}
