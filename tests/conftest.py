import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


@pytest.fixture(scope="session")
def synth_root(tmp_path_factory):
    from lacd.datasets import synthetic
    root = tmp_path_factory.mktemp("synth")
    synthetic.generate(root, "synthetic_assembly", n_train=24, n_val=12, n_test_good=8, n_per_anomaly=4, size=96, seed=1)
    return root
