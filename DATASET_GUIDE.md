# Dataset guide

## MVTec LOCO AD (primary)
Official source: https://www.mvtec.com/company/research/datasets/mvtec-loco (licence applies; do not redistribute). Expected layout: `<cat>/train/good`, `validation/good`, `test/{good,logical_anomalies,structural_anomalies}`, `ground_truth/<type>/<stem>/*.png`. Run `scripts/inspect_dataset.py` and compare the total with the proposal's 3,644.

## Local tabletop dataset (not yet collected)
| Aspect | Guidance |
|---|---|
| Rig | Fixed camera on a stand, top-down or fixed oblique angle, same height every session |
| Lighting | Diffuse, constant; avoid hard shadows and glare; no auto-exposure changes |
| Background | Plain matte surface; mark a fixed placement area |
| Resolution | ≥ 1024 px on the long side, PNG |
| Normal images | ≥ 100 per assembly with small natural placement variation |
| Anomalies | ≥ 15 each: `missing_component`, `extra_component`, `misplaced_component`, `structural_anomalies` |
| Masks | Binary PNGs in `ground_truth/<type>/<stem>/000.png` (optional but needed for pixel metrics) |
| Metadata | `annotations.json` (`subtype`), `assembly_spec.json` (expected components/positions), capture-session id in the file name |
| Splits | Group by session with `grouped_split` (70/15/15) so near-duplicates never cross splits |
Synthetic images (`scripts/prepare_dataset.py --synthetic`) are marked with `SYNTHETIC.txt` and must never be reported as real results.
