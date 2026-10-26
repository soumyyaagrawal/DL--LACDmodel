"""Run inference on one image with a saved checkpoint; writes JSON + heatmap + overlay PNGs."""
import _common  # noqa: F401
import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image

from lacd.datasets.loco import load_rgb
from lacd.models.detector import Detector
from lacd.visualization.overlay import heatmap_rgb, normalize_map, overlay

ap = argparse.ArgumentParser()
ap.add_argument("--checkpoint", required=True, help="e.g. artifacts/run/<category>/relation")
ap.add_argument("--image", required=True)
ap.add_argument("--out", default="results/predictions")
a = ap.parse_args()
det = Detector.load(a.checkpoint)
img = load_rgb(a.image)
p = det.predict(img)
m = p.pop("anomaly_map")
m01 = normalize_map(m, 0.0, max(p["pixel_threshold"] * 2, 1e-3))
out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
stem = Path(a.image).stem
Image.fromarray(heatmap_rgb(m01)).save(out / f"{stem}_heatmap.png")
Image.fromarray(overlay(img, m01)).save(out / f"{stem}_overlay.png")
(out / f"{stem}.json").write_text(json.dumps(p, indent=1, default=float))
print(json.dumps({k: p[k] for k in ("model_name", "image_level_anomaly_score", "predicted_status", "threshold_used",
                                    "explanation", "inference_latency_ms", "device", "warnings")}, indent=1, default=float))
