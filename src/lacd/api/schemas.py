from typing import Optional

from pydantic import BaseModel


class ModelInfo(BaseModel):
    model_id: str
    run: str
    category: str
    kind: str
    checkpoint_id: str
    weights_source: str
    smoke_only: bool
    threshold: float


class Region(BaseModel):
    proto: int
    cx: float
    cy: float
    w: float
    h: float
    area: float
    relation_z: Optional[float]
    appearance_z: float


class Explanation(BaseModel):
    text: str
    evidence: str
    kind: str


class PredictionResponse(BaseModel):
    model_id: str
    model_name: str
    checkpoint_id: str
    image_level_anomaly_score: float
    score_semantics: str
    predicted_status: str
    threshold_used: float
    appearance_z: float
    relation_z: float
    regions: list[Region]
    relation_violations: list[dict]
    explanation: Explanation
    confidence_or_evidence: str
    inference_latency_ms: float
    device: str
    warnings: list[str]
    image_size: list[int]
    heatmap_png_base64: str
    overlay_png_base64: str
    map_display_range: list[float]


class CompareResponse(BaseModel):
    baseline: PredictionResponse
    relation: PredictionResponse
