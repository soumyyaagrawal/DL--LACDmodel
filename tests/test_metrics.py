import numpy as np

from lacd.evaluation.metrics import image_metrics, pixel_metrics


def test_image_metrics_perfect_and_confusion():
    m = image_metrics([0, 0, 1, 1], [0.1, 0.2, 0.8, 0.9], 0.5)
    assert m["auroc"] == 1.0 and m["precision"] == 1.0 and m["recall"] == 1.0 and m["f1"] == 1.0
    assert m["confusion_matrix"] == [[2, 0], [0, 2]]


def test_image_metrics_single_class_auroc_none():
    assert image_metrics([0, 0], [0.1, 0.2], 0.5)["auroc"] is None


def test_pixel_metrics_known_values():
    g = np.zeros((10, 10), np.uint8); g[:5, :5] = 1
    perfect = g.astype(np.float32)
    r = pixel_metrics([perfect], [g], 0.5)
    assert r["pixel_auroc"] == 1.0 and r["iou_mean_anomalous"] == 1.0 and r["dice_mean_anomalous"] == 1.0
    half = np.zeros((10, 10), np.float32); half[:5, :2] = 1  # predicts 10 of 25 GT pixels
    r = pixel_metrics([half], [g], 0.5)
    assert abs(r["iou_mean_anomalous"] - 10 / 25) < 1e-9
    assert abs(r["dice_mean_anomalous"] - 2 * 10 / (10 + 25)) < 1e-9
