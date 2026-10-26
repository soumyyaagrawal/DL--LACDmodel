"""Turn RelationNet outputs into region violation scores, count violations and a relation map."""
from __future__ import annotations

import numpy as np
import torch
from scipy.ndimage import gaussian_filter

from lacd.relations.explain import evidence_level


@torch.no_grad()
def raw_relation_terms(net, tens: dict, device) -> dict:
    """Per-region violation v_i = node NLL + mean pair NLL to partners; per-prototype count z."""
    t = {k: v[None].to(device) for k, v in tens.items()}
    net.eval()
    out = net(t["proto"], t["app"], t["geom"], t["mask"], t["counts"])
    pv = out["pair_valid"][0]
    pair_mean = (out["pair_nll"][0] * pv).sum(1) / pv.sum(1).clamp(min=1)
    # a region is also a partner of others, so use symmetric mean (i->j and j->i)
    pair_mean_t = (out["pair_nll"][0] * pv).sum(0) / pv.sum(0).clamp(min=1)
    v = out["node_nll"][0] + 0.5 * (pair_mean + pair_mean_t)
    return {"v": v.cpu().numpy(), "mask": tens["mask"].numpy(), "count_z": out["count_z"][0].cpu().numpy(),
            "counts": tens["counts"].numpy(), "node": out["node_nll"][0].cpu().numpy()}


def relation_raw_score(terms: dict, region_mu: float, region_sd: float) -> tuple[float, np.ndarray]:
    z_reg = (terms["v"] - region_mu) / region_sd
    z_reg = np.where(terms["mask"], z_reg, -np.inf)
    zmax = float(z_reg.max()) if terms["mask"].any() else 0.0
    cmax = float(np.abs(terms["count_z"]).max())
    return max(zmax, cmax), z_reg


def find_violations(regions, z_reg, terms, proto_mean_pos, proto_mean_count, z_thr_region, z_thr_count) -> list[dict]:
    out: list[dict] = []
    cz, counts = terms["count_z"], terms["counts"]
    for k in range(len(cz)):
        if abs(cz[k]) < z_thr_count:
            continue
        if cz[k] < 0:
            out.append({"type": "missing", "proto": int(k), "observed": int(counts[k]),
                        "expected": float(proto_mean_count[k]), "z": float(abs(cz[k])),
                        "location": [float(proto_mean_pos[k][0]), float(proto_mean_pos[k][1])],
                        "evidence": evidence_level(abs(cz[k]), z_thr_count)})
        else:
            cand = [i for i, r in enumerate(regions) if r.proto == k]
            i = max(cand, key=lambda i: z_reg[i]) if cand else None
            out.append({"type": "extra", "proto": int(k), "observed": int(counts[k]),
                        "expected": float(proto_mean_count[k]), "z": float(cz[k]),
                        "location": [regions[i].cx, regions[i].cy] if i is not None else None,
                        "region_index": i, "evidence": evidence_level(cz[k], z_thr_count)})
    flagged_protos = {v["proto"] for v in out}
    for i, r in enumerate(regions[: len(z_reg)]):
        if np.isfinite(z_reg[i]) and z_reg[i] >= z_thr_region and r.proto not in flagged_protos:
            out.append({"type": "misplaced", "proto": int(r.proto), "region_index": i, "z": float(z_reg[i]),
                        "location": [r.cx, r.cy], "evidence": evidence_level(z_reg[i], z_thr_region)})
    out.sort(key=lambda v: -v["z"])
    return out


def relation_map(regions, z_reg, violations, size_hw, proto_mean_size, sigma_frac=0.03) -> np.ndarray:
    """Relation-violation map in z-units aligned to the (resized) image; missing → bump at expected location."""
    H, W = size_hw
    m = np.zeros((H, W), np.float32)

    def paint(cx, cy, w, h, val):
        x0, x1 = int(max(0, (cx - w / 2) * W)), int(min(W, (cx + w / 2) * W) + 1)
        y0, y1 = int(max(0, (cy - h / 2) * H)), int(min(H, (cy + h / 2) * H) + 1)
        m[y0:y1, x0:x1] = np.maximum(m[y0:y1, x0:x1], val)

    for i, r in enumerate(regions[: len(z_reg)]):
        if np.isfinite(z_reg[i]) and z_reg[i] > 0:
            paint(r.cx, r.cy, r.w, r.h, z_reg[i])
    for v in violations:
        if v["type"] == "missing" and v.get("location"):
            w, h = proto_mean_size[v["proto"]]
            paint(v["location"][0], v["location"][1], w, h, v["z"])
    return gaussian_filter(m, sigma_frac * max(H, W))
