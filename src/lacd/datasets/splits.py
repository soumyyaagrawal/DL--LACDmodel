"""Reproducible, group-aware split manifests.

For MVTec LOCO AD the OFFICIAL train/validation/test split is preserved (no test image moves to
training). The 70/15/15 split from the proposal is applied only to separately collected data
(local tabletop dataset), grouped by capture session to avoid near-duplicate leakage.
"""
from __future__ import annotations

import random
from collections import defaultdict

from lacd.datasets.loco import Sample


def official_manifest(samples: list[Sample], seed: int) -> dict:
    return {"protocol": "official", "seed": seed,
            "splits": {s.path: s.split for s in samples}}


def grouped_split(items: list[tuple[str, str]], ratios=(0.70, 0.15, 0.15), seed: int = 0) -> dict[str, str]:
    """items: (id, group). Whole groups are assigned to one split. Returns id -> split name."""
    assert abs(sum(ratios) - 1) < 1e-6
    groups: dict[str, list[str]] = defaultdict(list)
    for i, g in items:
        groups[g].append(i)
    keys = sorted(groups)
    random.Random(seed).shuffle(keys)
    n = len(items)
    targets = [r * n for r in ratios]
    names = ["train", "validation", "test"]
    filled = [0, 0, 0]
    out: dict[str, str] = {}
    for g in keys:
        # place group in the split with the largest remaining deficit
        k = max(range(3), key=lambda j: targets[j] - filled[j])
        for i in groups[g]:
            out[i] = names[k]
        filled[k] += len(groups[g])
    return out
