"""Fitting + validation-based calibration for both detectors. Test data is NEVER touched here."""
from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import DataLoader

from lacd.datasets.loco import LocoDataset, Sample, load_rgb, preprocess, select
from lacd.models.detector import Detector
from lacd.models.relation_net import RelationNet
from lacd.relations.featurize import RegionFeaturizer, stack
from lacd.relations.regions import PrototypeRegionExtractor
from lacd.relations.scoring import raw_relation_terms, relation_raw_score
from lacd.training.train_relation import train_relation_net


def _loader(samples: list[Sample], cfg) -> DataLoader:
    return DataLoader(LocoDataset(samples, cfg.data.image_size), batch_size=cfg.data.batch_size,
                      shuffle=False, num_workers=cfg.data.num_workers)


def train_samples(samples: list[Sample], cfg) -> list[Sample]:
    tr = select(samples, "train", only_good=True)
    if cfg.data.max_train_images:
        tr = tr[: cfg.data.max_train_images]
    if not tr:
        raise RuntimeError("No training images found (train/good is empty).")
    return tr


def _val_normals(samples: list[Sample]) -> list[Sample]:
    v = select(samples, "validation", only_good=True)
    if not v:
        raise RuntimeError("No validation/good images found; thresholds cannot be calibrated without them.")
    return v


def calibrate_appearance(det: Detector, val: list[Sample]) -> dict:
    S = det.cfg.data.image_size
    scores = []
    for s in val:
        scores.append(det.raw(preprocess(load_rgb(s.path), S))["app_img"])
    scores = np.array(scores)
    det.cal.update(app_mean=float(scores.mean()), app_std=float(scores.std() + 1e-6), n_val=len(val))
    return {"val_app_scores": scores.tolist(), "warn_small_val": len(val) < 20}


def finish_calibration(det: Detector, val: list[Sample]) -> dict:
    """Relation calibration (if applicable), then threshold & pixel threshold from val-normal combined outputs."""
    S, sc = det.cfg.data.image_size, det.cfg.score
    raws = [det.raw(preprocess(load_rgb(s.path), S)) for s in val]
    if det.kind == "relation":
        vs = np.concatenate([r["terms"]["v"][r["terms"]["mask"]] for r in raws])
        det.cal.update(region_mu=float(vs.mean()), region_sd=float(vs.std() + 1e-6))
        rel = np.array([relation_raw_score(r["terms"], det.cal["region_mu"], det.cal["region_sd"])[0] for r in raws])
        det.cal.update(rel_mean=float(rel.mean()), rel_std=float(rel.std() + 1e-6))
    fins = [det.finalize(r) for r in raws]
    scores = np.array([f["score"] for f in fins])
    det.cal["threshold"] = float(np.percentile(scores, sc.threshold_percentile))
    rng = np.random.default_rng(det.cfg.seed)
    pix = np.concatenate([rng.choice(f["map"].ravel(), min(20000, f["map"].size), replace=False) for f in fins])
    det.cal["pixel_threshold"] = float(np.percentile(pix, sc.pixel_threshold_percentile))
    det.cal["threshold_procedure"] = (f"{sc.threshold_percentile}th percentile of combined score over "
                                      f"{len(val)} validation-normal images")
    det.cal["pixel_threshold_procedure"] = (f"{sc.pixel_threshold_percentile}th percentile of map values over "
                                            "validation-normal pixels")
    return {"val_scores": scores.tolist()}


def fit_baseline(det: Detector, samples: list[Sample]) -> dict:
    tr, val = train_samples(samples, det.cfg), _val_normals(samples)
    info = det.baseline.fit(_loader(tr, det.cfg))
    calibrate_appearance(det, val)
    finish_calibration(det, val)
    info.update(n_train_images=len(tr), n_val_images=len(val))
    return info


def fit_relation(det: Detector, samples: list[Sample], out_dir: str, resume: bool = False) -> dict:
    """det must already have a fitted/loaded appearance memory (shared with the baseline)."""
    cfg = det.cfg
    tr, val = train_samples(samples, cfg), _val_normals(samples)
    # 1) patch features of normal training images -> prototypes
    grids = []
    for b in _loader(tr, cfg):
        f = det.extractor(b["image"].to(det.device)).permute(0, 2, 3, 1).cpu().numpy()
        grids.extend(list(f))
    det.regions_ex = PrototypeRegionExtractor(cfg.regions, cfg.seed)
    det.regions_ex.fit(np.concatenate([g.reshape(-1, g.shape[-1]) for g in grids]))
    # 2) regions (no appearance scores for train: memory bank contains these images)
    tr_regs = [det.regions_ex.regions(g) for g in grids]
    det.featurizer = RegionFeaturizer(cfg.regions.n_prototypes, cfg.regions.pca_dim, cfg.regions.max_regions)
    det.featurizer.fit(tr_regs)
    train_t = stack([det.featurizer.tensors(r) for r in tr_regs])
    val_grids = []
    for b in _loader(val, cfg):
        f = det.extractor(b["image"].to(det.device)).permute(0, 2, 3, 1).cpu().numpy()
        val_grids.extend(list(f))
    val_t = stack([det.featurizer.tensors(det.regions_ex.regions(g)) for g in val_grids])
    # 3) per-prototype normal statistics (for 'expected' location/count/size in explanations)
    K = cfg.regions.n_prototypes
    pos, cnt, size = np.zeros((K, 2)), train_t["counts"].numpy().mean(0), np.full((K, 2), 0.1)
    for k in range(K):
        rs = [r for rl in tr_regs for r in rl if r.proto == k]
        if rs:
            pos[k] = [np.mean([r.cx for r in rs]), np.mean([r.cy for r in rs])]
            size[k] = [np.mean([r.w for r in rs]), np.mean([r.h for r in rs])]
    det.cal.update(proto_mean_pos=pos, proto_mean_count=cnt, proto_mean_size=size)
    # 4) train relation net
    det.net = RelationNet(K, det.featurizer.d, cfg.relation.hidden)
    summary = train_relation_net(det.net, train_t, val_t, cfg.relation, det.device, out_dir, cfg.seed, resume)
    det.net.to(det.device).eval()
    # 5) calibration on validation normals (alpha/beta fixed from config; see EXPERIMENTS.md)
    finish_calibration(det, val)
    summary.update(n_train_images=len(tr), n_val_images=len(val), mean_regions_per_train_image=float(train_t["mask"].sum(1).float().mean()))
    return summary
