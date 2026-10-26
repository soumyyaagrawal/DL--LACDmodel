import torch


def get_device(pref: str = "auto") -> torch.device:
    """Resolve a device string; falls back to CPU when CUDA is requested but unavailable."""
    if pref == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if pref.startswith("cuda") and not torch.cuda.is_available():
        print("[warn] CUDA requested but not available; falling back to CPU.")
        return torch.device("cpu")
    return torch.device(pref)
