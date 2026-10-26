import io
import shutil
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image

from lacd.datasets.loco import discover


def _png(arr) -> bytes:
    b = io.BytesIO(); Image.fromarray(arr).save(b, "PNG"); return b.getvalue()


@pytest.fixture(scope="module")
def api(tmp_path_factory, synth_root):
    """Build artifacts quickly (tiny config) and return a TestClient bound to them."""
    import os
    from lacd.models.detector import Detector
    from lacd.training.fit import fit_baseline, fit_relation
    from lacd.utils.config import Config
    from lacd.utils.seed import set_seed
    art = tmp_path_factory.mktemp("art") / "smoke" / "synthetic_assembly"
    cfg = Config(**{"device": "cpu", "backbone": {"pretrained": False, "allow_random_init": True, "feature_dim": 32},
                    "data": {"root": str(synth_root), "category": "synthetic_assembly", "image_size": 96},
                    "baseline": {"coreset_ratio": 0.2, "proj_dim": 32}, "regions": {"n_prototypes": 5, "pca_dim": 4},
                    "relation": {"epochs": 5, "hidden": 16}})
    set_seed(0); s = discover(str(synth_root), "synthetic_assembly")
    b = Detector(cfg, "baseline"); fit_baseline(b, s); b.save(art / "baseline")
    r = Detector(cfg, "relation"); r.extractor.load_state_dict(b.extractor.state_dict())
    r.extractor.weights_source = "random_init"; r.baseline.load_state(b.baseline.state())
    r.cal.update({k: v for k, v in b.cal.items() if k.startswith("app_") or k == "n_val"})
    fit_relation(r, s, str(art / "relation")); r.save(art / "relation")
    os.environ.update(LACD_ARTIFACT_DIR=str(art.parent.parent), LACD_RESULTS_DIR=str(art.parent.parent / "none"),
                      LACD_DATA_ROOT=str(synth_root), LACD_LOCAL_DATA_ROOT=str(art.parent.parent / "nolocal"),
                      LACD_MAX_UPLOAD_MB="1", LACD_DEVICE="cpu")
    from lacd.api.main import create_app
    return TestClient(create_app()), s


def test_health_and_models(api):
    c, _ = api
    assert c.get("/api/health").json()["status"] == "ok"
    ms = c.get("/api/models").json()
    assert {m["kind"] for m in ms} == {"baseline", "relation"} and all(m["smoke_only"] for m in ms)


def test_predict_real_inference_and_alignment(api):
    c, s = api
    path = [x for x in s if x.label == 1][0].path
    img = np.asarray(Image.open(path).convert("RGB"))
    r = c.post("/api/predict", files={"file": ("a.png", _png(img), "image/png")},
               data={"model_id": "smoke/synthetic_assembly/relation"})
    assert r.status_code == 200, r.text
    j = r.json()
    assert j["image_size"] == [img.shape[1], img.shape[0]] and j["score_semantics"].startswith("calibrated")
    hm = Image.open(io.BytesIO(__import__("base64").b64decode(j["heatmap_png_base64"])))
    assert hm.size == (img.shape[1], img.shape[0])


def test_compare(api):
    c, s = api
    img = np.asarray(Image.open(s[0].path).convert("RGB"))
    r = c.post("/api/compare", files={"file": ("a.png", _png(img), "image/png")},
               data={"run": "smoke", "category": "synthetic_assembly"})
    assert r.status_code == 200 and r.json()["baseline"]["model_name"] == "baseline"


@pytest.mark.parametrize("name,content,ctype,code", [
    ("a.txt", b"hello", "text/plain", 415), ("a.png", b"not an image", "image/png", 400),
    ("big.png", b"0" * (2 * 2**20), "image/png", 413)])
def test_invalid_uploads(api, name, content, ctype, code):
    c, _ = api
    r = c.post("/api/predict", files={"file": (name, content, ctype)}, data={"model_id": "smoke/synthetic_assembly/baseline"})
    assert r.status_code == code


def test_unknown_model_and_path_traversal_rejected(api):
    c, s = api
    f = {"file": ("a.png", _png(np.zeros((8, 8, 3), np.uint8)), "image/png")}
    assert c.post("/api/predict", files=f, data={"model_id": "../../etc/passwd"}).status_code == 404


def test_missing_results_and_dataset_states(api):
    c, _ = api
    assert c.get("/api/evaluation/summary").json()["available"] is False
    d = c.get("/api/dataset/summary").json()
    assert d["mvtec_loco"]["available"] is True and d["local_tabletop"]["available"] is False
