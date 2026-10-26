# Implementation plan

| Phase | Deliverable | State |
|---|---|---|
| 1 Inspect | Environment findings: 1 CPU, 3.9 GB RAM, no GPU; PyPI reachable; mvtec.com, pytorch.org, huggingface.co unreachable | Done |
| 2 Data | LOCO loader, integrity report, grouped splits, synthetic generator | Done (tested on synthetic only) |
| 3 Baseline | Feature extractor, coreset memory bank, kNN scoring, save/load | Done |
| 4 Relation | Prototype regions, RelationNet, calibrated combination, rule-based explanations | Done (needs real-data tuning) |
| 5 Eval | Image/pixel metrics, per-type split, latency/memory, plots, CSV/JSON | Done; real runs pending |
| 6 API | FastAPI endpoints + tests | Done except `/api/dataset/validate` |
| 7 UI | Dashboard, Inspect/compare, Evaluation, Dataset, Local dataset (info only) | Done; upload workspace pending |
| 8–9 Verify/docs | Tests, build, headless UI run, README | Done for smoke path; ARCHITECTURE.md/EXPERIMENTS.md/final report pending |

Risks: no pretrained weights offline; LOCO validation has no anomalies (alpha/beta untunable); relation scoring unproven on real data.
Acceptance criteria: see the master prompt §15; unmet items are listed in README → Status.
