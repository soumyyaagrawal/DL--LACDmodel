"""MVTec LOCO AD discovery, annotation parsing, integrity report and torch Dataset.

Expected official layout (also used for the local tabletop dataset and the synthetic fixture):
  <root>/<category>/train/good/*.png
  <root>/<category>/validation/good/*.png
  <root>/<category>/test/{good,logical_anomalies,structural_anomalies}/*.png
  <root>/<category>/ground_truth/{logical_anomalies,structural_anomalies}/<image_stem>/*.png
Local datasets may use other anomaly folder names (e.g. missing_component); those are kept as-is.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field, asdict
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

IMG_EXT = {".png", ".jpg", ".jpeg", ".bmp"}
TYPE_ALIASES = {"logical_anomalies": "logical", "structural_anomalies": "structural", "good": "good"}


@dataclass
class Sample:
    path: str
    category: str
    split: str            # train | validation | test
    anomaly_type: str     # good | logical | structural | <local folder name>
    label: int            # 0 normal, 1 anomalous
    mask_paths: list[str] = field(default_factory=list)
    subtype: str | None = None  # finer label if an annotations.json provides it (e.g. "missing")

    @property
    def stem(self) -> str:
        return Path(self.path).stem


def list_categories(root: str | Path) -> list[str]:
    r = Path(root)
    if not r.is_dir():
        return []
    return sorted(p.name for p in r.iterdir() if p.is_dir() and (p / "train").is_dir())


def discover(root: str | Path, category: str) -> list[Sample]:
    """Find all samples for one category. Raises with an actionable message if missing."""
    cdir = Path(root) / category
    if not cdir.is_dir():
        raise FileNotFoundError(
            f"Category folder not found: {cdir}. Download MVTec LOCO AD from the official page "
            "(https://www.mvtec.com/company/research/datasets/mvtec-loco), extract it, and set data.root."
        )
    ann_file = cdir / "annotations.json"
    ann = json.loads(ann_file.read_text()) if ann_file.exists() else {}
    out: list[Sample] = []
    for split in ("train", "validation", "test"):
        sdir = cdir / split
        if not sdir.is_dir():
            continue
        for tdir in sorted(p for p in sdir.iterdir() if p.is_dir()):
            atype = TYPE_ALIASES.get(tdir.name, tdir.name)
            for img in sorted(p for p in tdir.iterdir() if p.suffix.lower() in IMG_EXT):
                masks: list[str] = []
                if atype != "good":
                    gt = cdir / "ground_truth" / tdir.name / img.stem
                    if gt.is_dir():
                        masks = sorted(str(m) for m in gt.iterdir() if m.suffix.lower() in IMG_EXT)
                    else:
                        single = cdir / "ground_truth" / tdir.name / f"{img.stem}_mask.png"
                        if single.exists():
                            masks = [str(single)]
                rel = f"{split}/{tdir.name}/{img.name}"
                out.append(Sample(str(img), category, split, atype, int(atype != "good"), masks,
                                  subtype=(ann.get(rel) or {}).get("subtype")))
    return out


def select(samples: list[Sample], split: str, only_good: bool | None = None) -> list[Sample]:
    s = [x for x in samples if x.split == split]
    if only_good is True:
        s = [x for x in s if x.label == 0]
    return s


def load_rgb(path: str) -> np.ndarray:
    return np.asarray(Image.open(path).convert("RGB"))


def load_mask(sample: Sample, size_hw: tuple[int, int] | None = None) -> np.ndarray | None:
    """Union of all binary masks as uint8 {0,1}; zeros for normal; None if anomalous w/o masks."""
    if sample.label == 0:
        return np.zeros(size_hw or load_rgb(sample.path).shape[:2], np.uint8)
    if not sample.mask_paths:
        return None
    m = None
    for p in sample.mask_paths:
        a = np.asarray(Image.open(p).convert("L")) > 0
        m = a if m is None else (m | a)
    m = m.astype(np.uint8)
    if size_hw and m.shape != tuple(size_hw):
        m = np.asarray(Image.fromarray(m * 255).resize((size_hw[1], size_hw[0]), Image.NEAREST)) > 0
        m = m.astype(np.uint8)
    return m


def preprocess(img: np.ndarray, size: int) -> torch.Tensor:
    """uint8 HxWx3 -> float [3,size,size] in [0,1] (bilinear resize; ImageNet norm is in the model)."""
    im = Image.fromarray(img).resize((size, size), Image.BILINEAR)
    return torch.from_numpy(np.asarray(im, dtype=np.float32) / 255.0).permute(2, 0, 1).contiguous()


class LocoDataset(Dataset):
    def __init__(self, samples: list[Sample], image_size: int, with_masks: bool = False):
        self.samples, self.size, self.with_masks = samples, image_size, with_masks

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        s = self.samples[i]
        img = load_rgb(s.path)
        item = {"image": preprocess(img, self.size), "label": s.label, "index": i,
                "orig_hw": torch.tensor(img.shape[:2])}
        if self.with_masks:
            m = load_mask(s, (self.size, self.size))
            item["mask"] = torch.from_numpy(m if m is not None else np.zeros((self.size,) * 2, np.uint8))
            item["has_mask"] = int(m is not None)
        return item


def integrity_report(root: str | Path, categories: list[str] | None = None) -> dict:
    """Counts by category/split/type, corrupt files, duplicates, size stats, mask alignment."""
    cats = categories or list_categories(root)
    rep: dict = {"root": str(root), "categories": {}, "totals": {"images": 0, "corrupt": 0, "duplicates": 0}}
    if not cats:
        rep["warning"] = "No categories found. Dataset not downloaded/extracted at this location."
        return rep
    for c in cats:
        samples = discover(root, c)
        counts: dict = {}
        sizes: dict = {}
        corrupt, hashes, dups, mask_bad, no_mask = [], {}, [], [], []
        for s in samples:
            key = f"{s.split}/{s.anomaly_type}"
            counts[key] = counts.get(key, 0) + 1
            try:
                with Image.open(s.path) as im:
                    im.verify()
                with Image.open(s.path) as im:
                    sizes[str(im.size)] = sizes.get(str(im.size), 0) + 1
                    wh = im.size
            except Exception as e:  # noqa: BLE001
                corrupt.append({"path": s.path, "error": str(e)})
                continue
            h = hashlib.sha1(Path(s.path).read_bytes()).hexdigest()
            if h in hashes:
                dups.append([hashes[h], s.path])
            else:
                hashes[h] = s.path
            if s.label == 1:
                if not s.mask_paths:
                    no_mask.append(s.path)
                for mp in s.mask_paths:
                    with Image.open(mp) as m:
                        if m.size != wh:
                            mask_bad.append({"mask": mp, "mask_size": m.size, "image_size": wh})
        rep["categories"][c] = {"counts": counts, "image_sizes": sizes, "corrupt": corrupt,
                                "duplicates": dups, "mask_size_mismatch": mask_bad,
                                "anomalous_without_mask": no_mask, "total": len(samples)}
        rep["totals"]["images"] += len(samples)
        rep["totals"]["corrupt"] += len(corrupt)
        rep["totals"]["duplicates"] += len(dups)
    return rep
