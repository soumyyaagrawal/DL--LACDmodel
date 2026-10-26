import cv2
import numpy as np


def normalize_map(m: np.ndarray, lo: float, hi: float) -> np.ndarray:
    """Display-only normalisation (does not alter stored scores)."""
    return np.clip((m - lo) / max(hi - lo, 1e-6), 0, 1)


def heatmap_rgb(m01: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(cv2.applyColorMap((m01 * 255).astype(np.uint8), cv2.COLORMAP_JET), cv2.COLOR_BGR2RGB)


def overlay(img: np.ndarray, m01: np.ndarray, alpha: float = 0.5) -> np.ndarray:
    h = heatmap_rgb(m01)
    return (img * (1 - alpha * m01[..., None]) + h * (alpha * m01[..., None])).astype(np.uint8)
