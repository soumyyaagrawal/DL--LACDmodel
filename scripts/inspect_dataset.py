"""Dataset integrity report: counts per category/split/type, corrupt files, duplicates, mask alignment."""
import _common  # noqa: F401
import argparse
import json
from pathlib import Path

from lacd.datasets.loco import integrity_report

ap = argparse.ArgumentParser()
ap.add_argument("--root", default="data/mvtec_loco_anomaly_detection")
ap.add_argument("--out", default="results/dataset_integrity.json")
a = ap.parse_args()
rep = integrity_report(a.root)
Path(a.out).parent.mkdir(parents=True, exist_ok=True)
Path(a.out).write_text(json.dumps(rep, indent=1))
if "warning" in rep:
    raise SystemExit(f"[error] {rep['warning']} (root={a.root})")
print(f"{'category':22s} {'images':>7s}  corrupt  dup  mask-mismatch  anomalous-without-mask")
for c, r in rep["categories"].items():
    print(f"{c:22s} {r['total']:7d} {len(r['corrupt']):8d} {len(r['duplicates']):4d} "
          f"{len(r['mask_size_mismatch']):14d} {len(r['anomalous_without_mask']):10d}")
    for k, v in sorted(r["counts"].items()):
        print(f"    {k:40s} {v}")
print("total images found:", rep["totals"]["images"], "(proposal states 3,644; verify against this count)")
print("report ->", a.out)
