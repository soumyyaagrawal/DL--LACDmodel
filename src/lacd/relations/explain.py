"""Rule-based explanations grounded ONLY in computed evidence (no generative text from pixels).

Violation types and the evidence required:
  missing    : prototype k observed count < learned mean and |count z| >= count_z_thr  (expected loc = train mean)
  extra      : prototype k observed count > learned mean and count z >= count_z_thr
  misplaced  : a region's node/pair NLL z-score >= region_z_thr while its prototype count is normal
  appearance : a region contains patches whose appearance z-score >= region_z_thr
Prototypes are unnamed appearance clusters, so messages say "component/region of type #k", never a real part name.
"""
from __future__ import annotations


def where(cx: float, cy: float) -> str:
    v = "upper" if cy < 1 / 3 else ("lower" if cy > 2 / 3 else "middle")
    h = "left" if cx < 1 / 3 else ("right" if cx > 2 / 3 else "center")
    if v == "middle" and h == "center":
        return "center"
    return f"{v}-{h}" if v != "middle" else h if h != "center" else "center"


def evidence_level(z: float, thr: float) -> str:
    if z >= 2 * thr:
        return "strong"
    if z >= thr:
        return "moderate"
    return "weak"


def build_explanation(status: str, app_z: float, rel_z: float, violations: list[dict],
                      app_regions: list[dict], z_thr: float) -> dict:
    if status == "normal":
        return {"text": "No anomaly evidence above the validation-calibrated threshold.",
                "evidence": "n/a", "kind": "normal"}
    lines: list[str] = []
    for v in violations:
        loc = where(*v["location"]) if v.get("location") else "unknown area"
        if v["type"] == "missing":
            lines.append(f"Expected {v['expected']:.1f} region(s) of type #{v['proto']} (learned from normal images) "
                         f"but observed {v['observed']}; a region is probably missing near the {loc} area "
                         f"[{v['evidence']} evidence, |z|={v['z']:.1f}].")
        elif v["type"] == "extra":
            lines.append(f"Observed {v['observed']} region(s) of type #{v['proto']} where about {v['expected']:.1f} "
                         f"are expected; an extra region is suspected near the {loc} area "
                         f"[{v['evidence']} evidence, z={v['z']:.1f}].")
        elif v["type"] == "misplaced":
            lines.append(f"A region of type #{v['proto']} in the {loc} area violates the learned spatial pattern "
                         f"(position/relation score z={v['z']:.1f}) [{v['evidence']} evidence].")
    seen: dict[str, float] = {}
    for r in app_regions:  # one line per distinct area (keep the strongest)
        w = where(r["cx"], r["cy"])
        seen[w] = max(seen.get(w, 0.0), r["z"])
    for w, z in sorted(seen.items(), key=lambda kv: -kv[1])[:2]:
        lines.append(f"Anomalous appearance in the {w} area (appearance z={z:.1f}).")
    if not lines:
        lines.append("The image-level score exceeds the threshold, but no specific missing/extra/misplaced "
                     "evidence was found; the cause is uncertain. See the heatmap for the most anomalous area.")
        kind, ev = "generic", "weak"
    else:
        kind = violations[0]["type"] if violations else "appearance"
        ev = max((v["evidence"] for v in violations), key=["weak", "moderate", "strong"].index) if violations else "moderate"
    return {"text": " ".join(lines), "evidence": ev, "kind": kind,
            "scores": {"appearance_z": round(app_z, 2), "relation_z": round(rel_z, 2)}}
