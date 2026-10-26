"""Write split manifests; optionally generate the SYNTHETIC smoke-test dataset.

  python scripts/prepare_dataset.py --root data/mvtec_loco_anomaly_detection --category breakfast_box
  python scripts/prepare_dataset.py --synthetic --root data/synthetic
"""
import _common  # noqa: F401
import argparse
import json
from pathlib import Path

from lacd.datasets import synthetic
from lacd.datasets.loco import discover
from lacd.datasets.splits import official_manifest

ap = argparse.ArgumentParser()
ap.add_argument("--root", default="data/mvtec_loco_anomaly_detection")
ap.add_argument("--category", default=None)
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--synthetic", action="store_true", help="generate SYNTHETIC data (smoke tests only)")
ap.add_argument("--out", default="results/split_manifest.json")
a = ap.parse_args()
if a.synthetic:
    p = synthetic.generate(a.root, seed=a.seed)
    print("generated SYNTHETIC dataset at", p, "(not a real benchmark)")
    a.category = a.category or p.name
if not a.category:
    raise SystemExit("[error] --category required")
samples = discover(a.root, a.category)
man = official_manifest(samples, a.seed)
counts: dict = {}
for s in samples:
    counts[f"{s.split}/{s.anomaly_type}"] = counts.get(f"{s.split}/{s.anomaly_type}", 0) + 1
man["counts"] = counts
man["note"] = ("Official LOCO split preserved; the 70/15/15 proposal split is applied only to separately collected "
               "data via lacd.datasets.splits.grouped_split (group = capture session).")
Path(a.out).parent.mkdir(parents=True, exist_ok=True)
Path(a.out).write_text(json.dumps(man, indent=1))
print(json.dumps(counts, indent=1)); print("manifest ->", a.out)
