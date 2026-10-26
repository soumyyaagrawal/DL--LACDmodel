"""Train the relation-aware model. Reuses the baseline's appearance memory (fit it first)."""
import _common  # noqa: F401
import argparse
import json
import time

from lacd.datasets.loco import discover
from lacd.models.detector import Detector
from lacd.training.fit import fit_relation
from lacd.utils.config import load_config
from lacd.utils.seed import set_seed

ap = argparse.ArgumentParser()
ap.add_argument("--config", required=True)
ap.add_argument("--category", default=None)
ap.add_argument("--root", default=None)
ap.add_argument("--out", default=None)
ap.add_argument("--resume", action="store_true")
a = ap.parse_args()
ov = {k: v for k, v in {"data.category": a.category, "data.root": a.root, "out_dir": a.out}.items() if v}
cfg = load_config(a.config, ov)
set_seed(cfg.seed)
base_dir = f"{cfg.out_dir}/{cfg.data.category}/baseline"
try:
    base = Detector.load(base_dir)
except FileNotFoundError as e:
    raise SystemExit(f"[error] {e}\nRun scripts/train_baseline.py with the same config first.")
det = Detector(cfg, "relation")
if base.extractor.weights_source == "random_init":
    det.extractor.load_state_dict(base.extractor.state_dict()); det.extractor.weights_source = "random_init"
det.baseline.load_state(base.baseline.state())
det.cal.update({k: v for k, v in base.cal.items() if k.startswith("app_") or k == "n_val"})
t0 = time.time()
samples = discover(cfg.data.root, cfg.data.category)
out = f"{cfg.out_dir}/{cfg.data.category}/relation"
info = fit_relation(det, samples, out, a.resume)
det.save(out, {"fit_info": info, "seconds": round(time.time() - t0, 1)})
print(json.dumps(info, indent=1)); print("checkpoint ->", out, det.checkpoint_id)
