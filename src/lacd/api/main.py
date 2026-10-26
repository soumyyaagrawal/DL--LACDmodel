"""FastAPI backend. Real inference only: if no checkpoint exists the API says so (HTTP 503/404)."""
from __future__ import annotations

import base64
import io
import json
import os
import threading
from pathlib import Path

import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image, UnidentifiedImageError

from lacd import __version__
from lacd.api.schemas import CompareResponse, ModelInfo, PredictionResponse
from lacd.datasets.loco import discover, list_categories
from lacd.models.detector import Detector
from lacd.visualization.overlay import heatmap_rgb, normalize_map, overlay

ALLOWED = {"image/png", "image/jpeg"}
MAX_PIXELS = 25_000_000


def _env(name: str, default: str) -> str:
    return os.environ.get(name, default)


class Registry:
    """Discovers checkpoints as <artifact_dir>/<run>/<category>/<kind>/meta.json; loads each once."""

    def __init__(self, root: Path):
        self.root, self._cache, self._lock = root, {}, threading.Lock()

    def list(self) -> list[ModelInfo]:
        out = []
        for meta in sorted(self.root.glob("*/*/*/meta.json")):
            try:
                m = json.loads(meta.read_text())
            except Exception:  # noqa: BLE001
                continue
            run, cat, kind = meta.parent.parent.parent.name, meta.parent.parent.name, meta.parent.name
            out.append(ModelInfo(model_id=f"{run}/{cat}/{kind}", run=run, category=cat, kind=kind,
                                 checkpoint_id=m["checkpoint_id"], weights_source=m["weights_source"],
                                 smoke_only=m["weights_source"] == "random_init" or run.startswith("smoke"),
                                 threshold=m["calibration"]["threshold"]))
        return out

    def get(self, model_id: str) -> Detector:
        if model_id not in {m.model_id for m in self.list()}:  # never trust client paths
            raise HTTPException(404, f"Unknown model_id '{model_id}'. See GET /api/models.")
        with self._lock:
            if model_id not in self._cache:
                self._cache[model_id] = Detector.load(self.root / model_id, _env("LACD_DEVICE", "auto"))
            return self._cache[model_id]


def _png64(arr: np.ndarray) -> str:
    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


def _decode(file: UploadFile, data: bytes, max_mb: int) -> np.ndarray:
    if file.content_type not in ALLOWED:
        raise HTTPException(415, f"Unsupported file type '{file.content_type}'. Use JPG or PNG.")
    if len(data) > max_mb * 2**20:
        raise HTTPException(413, f"File too large (limit {max_mb} MB).")
    try:
        im = Image.open(io.BytesIO(data))
        if im.width * im.height > MAX_PIXELS:
            raise HTTPException(413, "Image resolution too large.")
        return np.asarray(im.convert("RGB"))
    except HTTPException:
        raise
    except (UnidentifiedImageError, OSError) as e:
        raise HTTPException(400, f"Not a valid image: {e}") from e


def create_app() -> FastAPI:
    app = FastAPI(title="LACD API", version=__version__,
                  description="Logical assembly error detection: appearance baseline vs relation-aware model.")
    origins = [o for o in _env("LACD_CORS_ORIGINS", "http://localhost:5173").split(",") if o]
    app.add_middleware(CORSMiddleware, allow_origins=origins, allow_methods=["GET", "POST"], allow_headers=["*"])
    reg = Registry(Path(_env("LACD_ARTIFACT_DIR", "artifacts")))
    results_dir = Path(_env("LACD_RESULTS_DIR", "results"))
    max_mb = int(_env("LACD_MAX_UPLOAD_MB", "10"))
    infer_lock = threading.Lock()
    app.state.registry = reg

    def to_response(model_id: str, det: Detector, img: np.ndarray) -> PredictionResponse:
        with infer_lock:
            p = det.predict(img)
        m = p.pop("anomaly_map")
        lo, hi = 0.0, max(p["pixel_threshold"] * 2, float(np.percentile(m, 99.5)), 1e-3)
        m01 = normalize_map(m, lo, hi)  # display-only normalisation; stored scores are unchanged
        return PredictionResponse(
            model_id=model_id, image_size=[img.shape[1], img.shape[0]],
            heatmap_png_base64=_png64(heatmap_rgb(m01)), overlay_png_base64=_png64(overlay(img, m01)),
            map_display_range=[lo, hi], **{k: v for k, v in p.items() if k != "pixel_threshold"})

    @app.get("/api/health")
    def health():
        import torch
        models = reg.list()
        return {"status": "ok", "version": __version__, "torch": torch.__version__,
                "cuda_available": torch.cuda.is_available(), "n_models": len(models),
                "data_root_present": Path(_env("LACD_DATA_ROOT", "data/mvtec_loco_anomaly_detection")).is_dir()}

    @app.get("/api/models", response_model=list[ModelInfo])
    def models():
        return reg.list()

    @app.get("/api/dataset/categories")
    def categories():
        out = {}
        for name, root in (("mvtec_loco", _env("LACD_DATA_ROOT", "data/mvtec_loco_anomaly_detection")),
                           ("local_tabletop", _env("LACD_LOCAL_DATA_ROOT", "data/local_tabletop"))):
            out[name] = list_categories(root)
        return out

    @app.get("/api/dataset/summary")
    def dataset_summary():
        res = {}
        for name, root in (("mvtec_loco", _env("LACD_DATA_ROOT", "data/mvtec_loco_anomaly_detection")),
                           ("local_tabletop", _env("LACD_LOCAL_DATA_ROOT", "data/local_tabletop"))):
            cats = list_categories(root)
            if not cats:
                res[name] = {"available": False, "message": f"Dataset not found at configured location for '{name}'."}
                continue
            d = {}
            for c in cats:
                counts: dict = {}
                for s in discover(root, c):
                    k = f"{s.split}/{s.anomaly_type}"
                    counts[k] = counts.get(k, 0) + 1
                d[c] = counts
            res[name] = {"available": True, "categories": d, "total_images": sum(sum(v.values()) for v in d.values())}
        return res

    @app.post("/api/predict", response_model=PredictionResponse)
    async def predict(file: UploadFile = File(...), model_id: str = Form(...)):
        data = await file.read()
        img = _decode(file, data, max_mb)
        det = await run_in_threadpool(reg.get, model_id)
        return await run_in_threadpool(to_response, model_id, det, img)

    @app.post("/api/compare", response_model=CompareResponse)
    async def compare(file: UploadFile = File(...), run: str = Form(...), category: str = Form(...)):
        data = await file.read()
        img = _decode(file, data, max_mb)
        out = {}
        for kind in ("baseline", "relation"):
            mid = f"{run}/{category}/{kind}"
            det = await run_in_threadpool(reg.get, mid)
            out[kind] = await run_in_threadpool(to_response, mid, det, img)
        return CompareResponse(**out)

    def _load_results():
        items = []
        for f in sorted(results_dir.glob("*/*/metrics_*.json")):
            try:
                items.append((f, json.loads(f.read_text())))
            except Exception:  # noqa: BLE001
                continue
        return items

    @app.get("/api/evaluation/summary")
    def eval_summary():
        items = _load_results()
        if not items:
            return {"available": False, "message": "No evaluation results found. Run scripts/evaluate.py."}
        out = []
        for f, r in items:
            out.append({"experiment": f.parent.parent.name, "category": f.parent.name, "model": r["model"],
                        "label": r.get("label", ""), "synthetic_data": r.get("synthetic_data", False),
                        "weights_source": r.get("weights_source"), "overall": r["overall"],
                        "by_type": r["by_type"], "pixel": r["pixel"], "latency": r.get("latency"),
                        "checkpoint_id": r.get("checkpoint_id"), "threshold_procedure": r.get("threshold_procedure"),
                        "roc": r.get("roc")})
        return {"available": True, "results": out}

    @app.get("/api/evaluation/categories")
    def eval_categories():
        return sorted({f.parent.name for f, _ in _load_results()})

    @app.get("/api/experiments")
    def experiments():
        return [{"experiment": f.parent.parent.name, "category": f.parent.name, "model": r["model"],
                 "checkpoint_id": r.get("checkpoint_id"), "label": r.get("label", ""),
                 "synthetic_data": r.get("synthetic_data", False)} for f, r in _load_results()]

    return app


app = create_app()
