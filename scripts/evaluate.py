"""Evaluate baseline and relation-aware checkpoints on identical test samples; write JSON/CSV/plots."""
import _common  # noqa: F401
import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from lacd.datasets.loco import discover, load_rgb
from lacd.evaluation.evaluate import evaluate_detector
from lacd.models.detector import Detector
from lacd.visualization.overlay import normalize_map, overlay

ap = argparse.ArgumentParser()
ap.add_argument("--artifacts", default="artifacts/run")
ap.add_argument("--category", required=True)
ap.add_argument("--root", default="data/mvtec_loco_anomaly_detection")
ap.add_argument("--out", default="results")
ap.add_argument("--label", default="", help="free-text tag stored in the result (e.g. 'SMOKE-SYNTHETIC')")
a = ap.parse_args()
out = Path(a.out) / a.category
out.mkdir(parents=True, exist_ok=True)
samples = discover(a.root, a.category)
allres = {}
for kind in ("baseline", "relation"):
    d = Path(a.artifacts) / a.category / kind
    if not (d / "meta.json").exists():
        print(f"[skip] {kind}: no checkpoint at {d}"); continue
    det = Detector.load(d)
    r = evaluate_detector(det, samples)
    internal = r.pop("_internal")
    r["label"] = a.label
    r["synthetic_data"] = (Path(a.root) / a.category / "SYNTHETIC.txt").exists()
    allres[kind] = r
    (out / f"metrics_{kind}.json").write_text(json.dumps(r, indent=1))
    # qualitative panels: 2 best/worst anomalous
    rows = r["rows"]
    anom = [i for i, x in enumerate(rows) if x["label"] == 1]
    order = sorted(anom, key=lambda i: rows[i]["score"])
    pick = order[:2] + order[-2:]
    fig, ax = plt.subplots(2, len(pick), figsize=(3 * len(pick), 6))
    for j, i in enumerate(pick):
        img = load_rgb(rows[i]["path"]); m = internal["preds"][i]["map"]
        import cv2
        m = cv2.resize(m, (img.shape[1], img.shape[0]))
        ax[0, j].imshow(img); ax[0, j].set_title(f"{rows[i]['type']} s={rows[i]['score']:.1f}", fontsize=8)
        ax[1, j].imshow(overlay(img, normalize_map(m, 0, max(r['pixel_threshold'] * 2, 1e-3)))); ax[0, j].axis("off"); ax[1, j].axis("off")
    plt.tight_layout(); plt.savefig(out / f"qualitative_{kind}.png", dpi=110); plt.close()
    hist = Path(a.artifacts) / a.category / kind / "relation_history.json"
    if hist.exists():
        h = json.loads(hist.read_text())
        plt.figure(figsize=(5, 3)); plt.plot([x["train"] for x in h], label="train"); plt.plot([x["val"] for x in h], label="val")
        plt.xlabel("epoch"); plt.ylabel("loss"); plt.legend(); plt.tight_layout(); plt.savefig(out / "relation_loss.png", dpi=110); plt.close()
if allres:
    plt.figure(figsize=(4, 4))
    for k, r in allres.items():
        plt.plot(r["roc"]["fpr"], r["roc"]["tpr"], label=f"{k} (AUROC={r['overall']['auroc']})")
    plt.plot([0, 1], [0, 1], "k:"); plt.xlabel("FPR"); plt.ylabel("TPR"); plt.legend(fontsize=7); plt.tight_layout()
    plt.savefig(out / "roc.png", dpi=110); plt.close()
    with open(out / "summary.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["model", "subset", "n", "auroc", "precision", "recall", "f1", "pixel_auroc", "iou_mean", "dice_mean"])
        for k, r in allres.items():
            subsets = {"all": (r["overall"], r["pixel"]["overall"])}
            for t, m in r["by_type"].items():
                subsets[t] = (m, r["pixel"].get(t, {}))
            for t, m in r["by_subtype"].items():
                subsets[f"sub:{t}"] = (m, {})
            for name, (m, p) in subsets.items():
                w.writerow([k, name, m["n"], m["auroc"], round(m["precision"], 4), round(m["recall"], 4), round(m["f1"], 4),
                            p.get("pixel_auroc"), p.get("iou_mean_anomalous"), p.get("dice_mean_anomalous")])
    print(open(out / "summary.csv").read())
    for k, r in allres.items():
        print(k, "latency:", r["latency"].get("mean_ms"), "ms on", r["latency"].get("device"), "| peak GPU mem:",
              r["latency"].get("peak_gpu_mem_mb") or r["latency"].get("peak_gpu_mem_note"))
    print("results ->", out)
else:
    raise SystemExit("[error] no checkpoints found; train first.")
