"""Validated experiment configuration (pydantic) loaded from YAML."""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import yaml
from pydantic import BaseModel, ConfigDict, Field


class _Cfg(BaseModel):
    model_config = ConfigDict(extra="forbid")


class BackboneCfg(_Cfg):
    name: str = "resnet18"
    layers: list[str] = ["layer2", "layer3"]
    pretrained: bool = True
    # Smoke tests only: allow random init when pretrained weights cannot be downloaded.
    allow_random_init: bool = False
    weights_path: Optional[str] = None  # local state_dict (.pth) overrides download
    feature_dim: Optional[int] = 128  # channel reduction (adaptive avg pool); None = keep


class DataCfg(_Cfg):
    root: str = "data/mvtec_loco_anomaly_detection"
    category: str = "breakfast_box"
    image_size: int = 256
    batch_size: int = 8
    num_workers: int = 0
    max_train_images: Optional[int] = None  # smoke-test cap; None = all


class BaselineCfg(_Cfg):
    coreset_ratio: float = 0.1
    proj_dim: int = 128
    k: int = 1
    blur_sigma: float = 4.0


class RegionCfg(_Cfg):
    n_prototypes: int = 8
    min_area_patches: int = 2
    pca_dim: int = 8
    max_regions: int = 24
    kmeans_samples: int = 50000


class RelationTrainCfg(_Cfg):
    epochs: int = 100
    lr: float = 2e-3
    weight_decay: float = 1e-4
    batch_size: int = 16
    hidden: int = 32
    lambda_node: float = 1.0   # L_node  : region geometry given appearance + context
    lambda_rel: float = 1.0    # L_pair  : pairwise displacement (distance / direction / ordering)
    lambda_cnt: float = 1.0    # L_count : per-prototype component counts
    translation_jitter: float = 0.03
    patience: int = 15
    scheduler: str = "cosine"
    min_lr: float = 1e-5


class ScoreCfg(_Cfg):
    alpha: float = 1.0  # weight of calibrated appearance score
    beta: float = 1.0   # weight of calibrated relation score
    threshold_percentile: float = 99.0  # of validation-normal combined scores
    count_z_thr: float = 3.0
    region_z_thr: float = 3.0
    pixel_threshold_percentile: float = 99.9


class Config(_Cfg):
    seed: int = 0
    device: str = "auto"
    out_dir: str = "artifacts/run"
    backbone: BackboneCfg = Field(default_factory=BackboneCfg)
    data: DataCfg = Field(default_factory=DataCfg)
    baseline: BaselineCfg = Field(default_factory=BaselineCfg)
    regions: RegionCfg = Field(default_factory=RegionCfg)
    relation: RelationTrainCfg = Field(default_factory=RelationTrainCfg)
    score: ScoreCfg = Field(default_factory=ScoreCfg)


def load_config(path: str | Path, overrides: dict | None = None) -> Config:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Config file not found: {p}")
    raw = yaml.safe_load(p.read_text()) or {}
    for dotted, val in (overrides or {}).items():
        cur = raw
        parts = dotted.split(".")
        for k in parts[:-1]:
            cur = cur.setdefault(k, {})
        cur[parts[-1]] = val
    return Config(**raw)
