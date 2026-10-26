export interface ModelInfo { model_id: string; run: string; category: string; kind: string; checkpoint_id: string; weights_source: string; smoke_only: boolean; threshold: number }
export interface Violation { type: string; proto: number; z: number; evidence: string; location?: number[] | null; observed?: number; expected?: number }
export interface Prediction {
  model_id: string; model_name: string; checkpoint_id: string; image_level_anomaly_score: number; score_semantics: string;
  predicted_status: string; threshold_used: number; appearance_z: number; relation_z: number;
  regions: { proto: number; cx: number; cy: number; w: number; h: number; area: number; relation_z: number | null; appearance_z: number }[];
  relation_violations: Violation[]; explanation: { text: string; evidence: string; kind: string };
  confidence_or_evidence: string; inference_latency_ms: number; device: string; warnings: string[];
  image_size: number[]; heatmap_png_base64: string; overlay_png_base64: string; map_display_range: number[];
}
export interface Compare { baseline: Prediction; relation: Prediction }
export interface Metrics { n: number; n_anomalous: number; n_normal: number; auroc: number | null; precision: number; recall: number; f1: number; confusion_matrix: number[][]; threshold: number }
export interface EvalResult {
  experiment: string; category: string; model: string; label: string; synthetic_data: boolean; weights_source: string;
  overall: Metrics; by_type: Record<string, Metrics>; pixel: Record<string, { pixel_auroc?: number | null; iou_mean_anomalous?: number | null; dice_mean_anomalous?: number | null }>;
  latency?: { mean_ms?: number; device?: string; peak_gpu_mem_mb?: number | null; peak_gpu_mem_note?: string | null; methodology?: string };
  checkpoint_id: string; threshold_procedure: string; roc?: { fpr: number[]; tpr: number[] };
}

export class ApiError extends Error {}

async function j<T>(res: Response): Promise<T> {
  if (!res.ok) {
    let msg = res.statusText;
    try { const b = await res.json(); msg = typeof b.detail === "string" ? b.detail : JSON.stringify(b.detail); } catch { /* keep statusText */ }
    throw new ApiError(`${res.status}: ${msg}`);
  }
  return res.json() as Promise<T>;
}
const get = <T,>(u: string) => fetch(u).then((r) => j<T>(r)).catch((e) => { throw e instanceof ApiError ? e : new ApiError("Backend unreachable. Is the API running on port 8000?"); });

export const api = {
  health: () => get<{ status: string; version: string; torch: string; cuda_available: boolean; n_models: number; data_root_present: boolean }>("/api/health"),
  models: () => get<ModelInfo[]>("/api/models"),
  datasetSummary: () => get<Record<string, { available: boolean; message?: string; total_images?: number; categories?: Record<string, Record<string, number>> }>>("/api/dataset/summary"),
  evalSummary: () => get<{ available: boolean; message?: string; results?: EvalResult[] }>("/api/evaluation/summary"),
  predict: (file: File, modelId: string) => {
    const f = new FormData(); f.append("file", file); f.append("model_id", modelId);
    return fetch("/api/predict", { method: "POST", body: f }).then((r) => j<Prediction>(r));
  },
  compare: (file: File, run: string, category: string) => {
    const f = new FormData(); f.append("file", file); f.append("run", run); f.append("category", category);
    return fetch("/api/compare", { method: "POST", body: f }).then((r) => j<Compare>(r));
  },
};
