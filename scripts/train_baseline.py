"""Fit the PatchCore-style appearance baseline and calibrate on validation normals."""
import _common  # noqa: F401
import argparse
import json
import time

from lacd.datasets.loco import discover
from lacd.models.detector import Detector
from lacd.training.fit import fit_baseline
from lacd.utils.config import load_config
from lacd.utils.seed import set_seed

ap = argparse.ArgumentParser()
ap.add_argument("--config", required=True)
ap.add_argument("--category", default=None)
ap.add_argument("--root", default=None)
ap.add_argument("--out", default=None)
a = ap.parse_args()
ov = {k: v for k, v in {"data.category": a.category, "data.root": a.root, "out_dir": a.out}.items() if v}
cfg = load_config(a.config, ov)
set_seed(cfg.seed)
t0 = time.time()
det = Detector(cfg, "baseline")
print(f"[baseline] device={det.device} backbone={cfg.backbone.name} weights={det.extractor.weights_source}")
samples = discover(cfg.data.root, cfg.data.category)
info = fit_baseline(det, samples)
out = f"{cfg.out_dir}/{cfg.data.category}/baseline"
det.save(out, {"fit_info": info, "seconds": round(time.time() - t0, 1)})
print(json.dumps(info, indent=1)); print("calibration:", {k: v for k, v in det.cal.items() if not hasattr(v, "shape")})
print("checkpoint ->", out, det.checkpoint_id)
