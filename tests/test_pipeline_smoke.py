"""End-to-end SMOKE test on SYNTHETIC data with a randomly initialised backbone.
Verifies plumbing (fit -> calibrate -> save/load -> predict -> evaluate). Says nothing about benchmark quality."""
import numpy as np
import pytest

from lacd.datasets.loco import discover, load_rgb
from lacd.evaluation.evaluate import evaluate_detector
from lacd.models.detector import Detector
from lacd.training.fit import fit_baseline, fit_relation
from lacd.utils.config import Config
from lacd.utils.seed import set_seed


@pytest.fixture(scope="module")
def trained(synth_root, tmp_path_factory):
    out = tmp_path_factory.mktemp("art")
    cfg = Config(**{"seed": 0, "device": "cpu", "out_dir": str(out),
                    "backbone": {"pretrained": False, "allow_random_init": True, "feature_dim": 32},
                    "data": {"root": str(synth_root), "category": "synthetic_assembly", "image_size": 96, "batch_size": 8},
                    "baseline": {"coreset_ratio": 0.2, "proj_dim": 32, "blur_sigma": 1.0},
                    "regions": {"n_prototypes": 5, "pca_dim": 4, "max_regions": 16, "kmeans_samples": 5000},
                    "relation": {"epochs": 15, "hidden": 16, "patience": 5}})
    set_seed(0)
    samples = discover(str(synth_root), "synthetic_assembly")
    base = Detector(cfg, "baseline"); fit_baseline(base, samples); base.save(out / "baseline")
    rel = Detector(cfg, "relation")
    rel.extractor.load_state_dict(base.extractor.state_dict()); rel.extractor.weights_source = "random_init"
    rel.baseline.load_state(base.baseline.state())
    rel.cal.update({k: v for k, v in base.cal.items() if k.startswith("app_") or k == "n_val"})
    fit_relation(rel, samples, str(out / "relation")); rel.save(out / "relation")
    return out, samples, cfg


def test_checkpoint_roundtrip_identical_predictions(trained):
    out, samples, _ = trained
    img = load_rgb([s for s in samples if s.split == "test"][0].path)
    for kind in ("baseline", "relation"):
        a = Detector.load(out / kind, device="cpu").predict(img)
        b = Detector.load(out / kind, device="cpu").predict(img)
        assert a["image_level_anomaly_score"] == pytest.approx(b["image_level_anomaly_score"])
        assert a["anomaly_map"].shape == img.shape[:2]  # map aligned to ORIGINAL image size
        assert a["predicted_status"] in ("normal", "anomalous") and a["checkpoint_id"].startswith(kind)
        assert "random_init" in " ".join(a["warnings"])


def test_relation_prediction_fields(trained):
    out, samples, _ = trained
    p = Detector.load(out / "relation", device="cpu").predict(load_rgb([s for s in samples if s.label == 1][0].path))
    for k in ("regions", "relation_violations", "explanation", "inference_latency_ms", "threshold_used"):
        assert k in p
    assert p["regions"] and {"cx", "cy", "proto"} <= set(p["regions"][0])


def test_evaluate_same_samples_and_metrics_present(trained):
    out, samples, _ = trained
    rs = {k: evaluate_detector(Detector.load(out / k, device="cpu"), samples) for k in ("baseline", "relation")}
    assert [r["path"] for r in rs["baseline"]["rows"]] == [r["path"] for r in rs["relation"]["rows"]]
    for r in rs.values():
        assert 0.0 <= r["overall"]["auroc"] <= 1.0 and "logical" in r["by_type"] and "structural" in r["by_type"]
        assert r["pixel"]["overall"]["pixel_auroc"] is not None
        assert r["latency"]["peak_gpu_mem_mb"] is None  # CPU run
        assert all(np.isfinite(x["score"]) for x in r["rows"])


def test_missing_checkpoint_error(tmp_path):
    with pytest.raises(FileNotFoundError):
        Detector.load(tmp_path / "none")
