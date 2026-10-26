"""Unified detector: appearance baseline (kind='baseline') or relation-aware (kind='relation').

The relation-aware detector REUSES the baseline's appearance score (same memory bank) and adds
region/relation reasoning, so the comparison isolates the effect of spatial reasoning.
combined_z = alpha * z_appearance + beta * z_relation  (each z calibrated on validation NORMALS).
"""
from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

import cv2
import numpy as np
import torch

from lacd import __version__
from lacd.datasets.loco import preprocess
from lacd.features.backbone import FeatureExtractor
from lacd.models.baseline import PatchCoreBaseline
from lacd.models.relation_net import RelationNet
from lacd.relations.explain import build_explanation, evidence_level
from lacd.relations.featurize import RegionFeaturizer
from lacd.relations.regions import PrototypeRegionExtractor
from lacd.relations.scoring import find_violations, raw_relation_terms, relation_map, relation_raw_score
from lacd.utils.config import BackboneCfg, Config
from lacd.utils.device import get_device


class Detector:
    def __init__(self, cfg: Config, kind: str = "baseline", device: torch.device | None = None):
        assert kind in ("baseline", "relation")
        self.cfg, self.kind = cfg, kind
        self.device = device or get_device(cfg.device)
        self.extractor = FeatureExtractor(cfg.backbone)
        self.baseline = PatchCoreBaseline(self.extractor, cfg.baseline, self.device, cfg.seed)
        self.regions_ex: PrototypeRegionExtractor | None = None
        self.featurizer: RegionFeaturizer | None = None
        self.net: RelationNet | None = None
        self.cal: dict = {}
        self.checkpoint_id: str | None = None

    # ------------------------------------------------------------------ raw (uncalibrated) pass
    @torch.no_grad()
    def raw(self, x: torch.Tensor) -> dict:
        """x [3,S,S] float in [0,1]. Returns uncalibrated components (maps at S x S)."""
        S = x.shape[-1]
        img, amap, ps, f = self.baseline.score(x[None], out_size=(S, S))
        r = {"app_img": float(img[0]), "app_map": amap[0], "size": S}
        if self.kind == "relation":
            grid = f[0].cpu().numpy()
            regs = self.regions_ex.regions(grid, ps[0])
            tens = self.featurizer.tensors(regs)
            r.update(regions=regs, terms=raw_relation_terms(self.net, tens, self.device))
        return r

    # ------------------------------------------------------------------ calibrated result
    def finalize(self, r: dict) -> dict:
        c, sc = self.cal, self.cfg.score
        a_img = (r["app_img"] - c["app_mean"]) / c["app_std"]
        a_map = (r["app_map"] - c["app_mean"]) / c["app_std"]
        out = {"app_z": a_img, "rel_z": 0.0, "violations": [], "regions": [], "app_regions": []}
        if self.kind == "baseline":
            out["score"], out["map"] = a_img, a_map
        else:
            raw_rel, z_reg = relation_raw_score(r["terms"], c["region_mu"], c["region_sd"])
            rel_z = (raw_rel - c["rel_mean"]) / c["rel_std"]
            viol = find_violations(r["regions"], z_reg, r["terms"], c["proto_mean_pos"], c["proto_mean_count"],
                                   sc.region_z_thr, sc.count_z_thr)
            rm = relation_map(r["regions"], z_reg, viol, (r["size"], r["size"]), c["proto_mean_size"])
            rm_z = np.where(rm > 0, (rm - c["rel_mean"]) / c["rel_std"], 0.0).astype(np.float32)
            out["rel_z"] = rel_z
            out["score"] = sc.alpha * a_img + sc.beta * rel_z
            out["map"] = sc.alpha * a_map + sc.beta * rm_z
            out["violations"] = viol
            out["rel_map"] = rm_z
            S = r["size"]
            for i, g in enumerate(r["regions"][: len(z_reg)]):
                az = (g.app_patch_max - c["app_mean"]) / c["app_std"]
                out["regions"].append({"proto": g.proto, "cx": g.cx, "cy": g.cy, "w": g.w, "h": g.h,
                                       "area": g.area, "relation_z": float(z_reg[i]) if np.isfinite(z_reg[i]) else None,
                                       "appearance_z": float(az)})
                if az >= sc.region_z_thr:
                    out["app_regions"].append({"cx": g.cx, "cy": g.cy, "z": float(az)})
        out["app_map_z"] = a_map
        return out

    # ------------------------------------------------------------------ public inference
    def predict(self, img: np.ndarray) -> dict:
        """img uint8 HxWx3. Returns a prediction dict; maps are resized to the ORIGINAL image size."""
        if not self.cal:
            raise RuntimeError("Detector is not calibrated/loaded; fit or load a checkpoint first.")
        H, W = img.shape[:2]
        t0 = time.perf_counter()
        x = preprocess(img, self.cfg.data.image_size)
        res = self.finalize(self.raw(x))
        if self.device.type == "cuda":
            torch.cuda.synchronize()
        ms = (time.perf_counter() - t0) * 1000
        thr = self.cal["threshold"]
        status = "anomalous" if res["score"] >= thr else "normal"
        expl = build_explanation(status, res["app_z"], res["rel_z"], res["violations"], res["app_regions"],
                                 self.cfg.score.region_z_thr)
        if self.kind == "baseline" and status == "anomalous":
            expl = {"text": "Appearance baseline: an anomalous region was detected (see heatmap). The baseline has no "
                            "spatial-relation reasoning, so it cannot say whether a component is missing, extra or misplaced.",
                    "evidence": "moderate" if res["score"] >= 2 * thr else "weak", "kind": "appearance_only"}
        amap = cv2.resize(res["map"], (W, H), interpolation=cv2.INTER_LINEAR)
        return {"model_name": self.kind, "checkpoint_id": self.checkpoint_id,
                "image_level_anomaly_score": float(res["score"]), "score_semantics": "calibrated z-score (not a probability)",
                "predicted_status": status, "threshold_used": float(thr), "anomaly_map": amap.astype(np.float32),
                "pixel_threshold": float(self.cal.get("pixel_threshold", thr)),
                "appearance_z": float(res["app_z"]), "relation_z": float(res["rel_z"]),
                "regions": res["regions"], "relation_violations": res["violations"], "explanation": expl,
                "confidence_or_evidence": expl["evidence"], "inference_latency_ms": ms, "device": str(self.device),
                "warnings": ([f"backbone weights: {self.extractor.weights_source}"]
                             if self.extractor.weights_source == "random_init" else [])}

    # ------------------------------------------------------------------ persistence
    def save(self, out_dir: str | Path, extra_meta: dict | None = None) -> Path:
        d = Path(out_dir)
        d.mkdir(parents=True, exist_ok=True)
        torch.save(self.baseline.state(), d / "appearance.pt")
        files = [d / "appearance.pt"]
        if self.extractor.weights_source == "random_init":
            torch.save(self.extractor.state_dict(), d / "backbone.pt")
        if self.kind == "relation":
            torch.save({"net": self.net.state_dict(), "featurizer": self.featurizer.state(),
                        "centers": torch.tensor(self.regions_ex.centers)}, d / "relation.pt")
            files.append(d / "relation.pt")
        h = hashlib.sha1()
        for f in files:
            h.update(f.read_bytes())
        self.checkpoint_id = f"{self.kind}-{h.hexdigest()[:10]}"
        meta = {"kind": self.kind, "checkpoint_id": self.checkpoint_id, "lacd_version": __version__,
                "torch": torch.__version__, "config": self.cfg.model_dump(), "calibration": _jsonable(self.cal),
                "weights_source": self.extractor.weights_source, **(extra_meta or {})}
        (d / "meta.json").write_text(json.dumps(meta, indent=1))
        return d

    @classmethod
    def load(cls, d: str | Path, device: str | None = None) -> "Detector":
        d = Path(d)
        if not (d / "meta.json").exists():
            raise FileNotFoundError(f"No checkpoint at {d} (meta.json missing). Train a model first.")
        meta = json.loads((d / "meta.json").read_text())
        cfg = Config(**meta["config"])
        if device:
            cfg.device = device
        if (d / "backbone.pt").exists():  # random-init smoke backbone: restore exact weights
            cfg.backbone.pretrained, cfg.backbone.allow_random_init = False, True
        det = cls(cfg, meta["kind"])
        if (d / "backbone.pt").exists():
            det.extractor.load_state_dict(torch.load(d / "backbone.pt", map_location="cpu", weights_only=True))
            det.extractor.weights_source = "random_init"
        det.baseline.load_state(torch.load(d / "appearance.pt", map_location="cpu", weights_only=True))
        if det.kind == "relation":
            ck = torch.load(d / "relation.pt", map_location="cpu", weights_only=True)
            det.featurizer = RegionFeaturizer.from_state(ck["featurizer"])
            det.regions_ex = PrototypeRegionExtractor(cfg.regions, cfg.seed)
            det.regions_ex.centers = ck["centers"].numpy()
            det.net = RelationNet(cfg.regions.n_prototypes, det.featurizer.d, cfg.relation.hidden)
            det.net.load_state_dict(ck["net"])
            det.net.to(det.device).eval()
        det.cal = _from_json(meta["calibration"])
        det.checkpoint_id = meta["checkpoint_id"]
        return det


def _jsonable(o):
    if isinstance(o, dict):
        return {k: _jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_jsonable(v) for v in o]
    if isinstance(o, np.ndarray):
        return {"__nd__": o.tolist()}
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    return o


def _from_json(o):
    if isinstance(o, dict):
        if "__nd__" in o:
            return np.array(o["__nd__"], dtype=np.float32)
        return {k: _from_json(v) for k, v in o.items()}
    if isinstance(o, list):
        return [_from_json(v) for v in o]
    return o
