"""Deterministic checks of the relation scoring / explanation logic (synthetic fixtures; NOT benchmarks)."""
import numpy as np

from lacd.relations.explain import build_explanation, where
from lacd.relations.regions import Region
from lacd.relations.scoring import find_violations, relation_map


def R(proto, cx, cy):
    return Region(proto, cx, cy, 0.1, 0.1, 0.01, np.zeros(4))


def terms(counts, count_z, nreg):
    return {"counts": np.array(counts, float), "count_z": np.array(count_z, float), "mask": np.ones(nreg, bool),
            "v": np.zeros(nreg), "node": np.zeros(nreg)}


MEAN_POS = [[0.2, 0.3], [0.7, 0.3]]
MEAN_CNT = [1.0, 1.0]


def test_normal_arrangement_no_violation():
    regs = [R(0, .2, .3), R(1, .7, .3)]
    v = find_violations(regs, np.array([0.1, 0.2]), terms([1, 1], [0, 0], 2), MEAN_POS, MEAN_CNT, 3, 3)
    assert v == []


def test_removed_region_flagged_missing_at_expected_location():
    regs = [R(1, .7, .3)]
    v = find_violations(regs, np.array([0.1]), terms([0, 1], [-5, 0], 1), MEAN_POS, MEAN_CNT, 3, 3)
    assert [x["type"] for x in v] == ["missing"] and v[0]["proto"] == 0 and v[0]["location"] == [0.2, 0.3]
    assert v[0]["evidence"] == "moderate"


def test_added_region_flagged_extra():
    regs = [R(0, .2, .3), R(1, .7, .3), R(1, .5, .5)]
    v = find_violations(regs, np.array([0.1, 0.1, 6.0]), terms([1, 2], [0, 7], 3), MEAN_POS, MEAN_CNT, 3, 3)
    assert v[0]["type"] == "extra" and v[0]["region_index"] == 2 and v[0]["evidence"] == "strong"


def test_shifted_region_flagged_misplaced_only_when_count_normal():
    regs = [R(0, .8, .8), R(1, .7, .3)]
    v = find_violations(regs, np.array([9.0, 0.1]), terms([1, 1], [0, 0], 2), MEAN_POS, MEAN_CNT, 3, 3)
    assert [x["type"] for x in v] == ["misplaced"] and v[0]["region_index"] == 0


def test_relation_map_aligned_and_peaks_at_violation():
    regs = [R(0, .8, .2)]
    m = relation_map(regs, np.array([8.0]), [], (64, 64), [(0.1, 0.1)])
    assert m.shape == (64, 64)
    y, x = np.unravel_index(m.argmax(), m.shape)
    assert abs(x / 64 - .8) < .1 and abs(y / 64 - .2) < .1


def test_explanation_grounded_and_uncertain_fallback():
    assert build_explanation("normal", 0, 0, [], [], 3)["kind"] == "normal"
    e = build_explanation("anomalous", 5, 0, [], [], 3)
    assert e["evidence"] == "weak" and "uncertain" in e["text"]
    assert where(0.9, 0.1) == "upper-right"


def test_duplicate_appearance_areas_are_merged():
    e = build_explanation("anomalous", 9, 0, [], [{"cx": .5, "cy": .5, "z": 10}, {"cx": .52, "cy": .48, "z": 14}], 3)
    assert e["text"].count("center area") == 1 and "14.0" in e["text"]
