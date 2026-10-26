from lacd.datasets.loco import discover, integrity_report, list_categories, load_mask, load_rgb, preprocess, select
from lacd.datasets.splits import grouped_split


def test_discovery_and_types(synth_root):
    assert list_categories(synth_root) == ["synthetic_assembly"]
    s = discover(synth_root, "synthetic_assembly")
    types = {(x.split, x.anomaly_type) for x in s}
    assert ("train", "good") in types and ("test", "logical") in types and ("test", "structural") in types
    assert all(x.label == (x.anomaly_type != "good") for x in s)
    assert {x.subtype for x in s if x.subtype} == {"missing", "extra", "misplaced", "defect"}


def test_rgb_and_mask_alignment(synth_root):
    s = [x for x in discover(synth_root, "synthetic_assembly") if x.label == 1][0]
    img = load_rgb(s.path)
    assert img.ndim == 3 and img.shape[2] == 3
    m = load_mask(s, img.shape[:2])
    assert m.shape == img.shape[:2] and m.sum() > 0
    assert load_mask(s, (48, 48)).shape == (48, 48)
    assert tuple(preprocess(img, 64).shape) == (3, 64, 64)


def test_normal_mask_is_zero_and_missing_dir_error(synth_root):
    s = select(discover(synth_root, "synthetic_assembly"), "test", only_good=True)[0]
    assert load_mask(s, (32, 32)).sum() == 0
    import pytest
    with pytest.raises(FileNotFoundError):
        discover(synth_root, "nope")


def test_integrity_report(synth_root):
    r = integrity_report(synth_root)
    c = r["categories"]["synthetic_assembly"]
    assert c["corrupt"] == [] and c["mask_size_mismatch"] == [] and c["anomalous_without_mask"] == []
    assert r["totals"]["images"] == c["total"]
    assert integrity_report(synth_root / "missing")["categories"] == {}


def test_grouped_split_no_group_leak():
    items = [(f"i{k}", f"g{k // 4}") for k in range(100)]
    a = grouped_split(items, seed=3)
    assert a == grouped_split(items, seed=3)  # reproducible
    by_group = {}
    for i, g in items:
        by_group.setdefault(g, set()).add(a[i])
    assert all(len(v) == 1 for v in by_group.values())
