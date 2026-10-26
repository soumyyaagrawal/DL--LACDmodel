import { useEffect, useMemo, useState } from "react";
import { api, Compare, ModelInfo, Prediction } from "../api";
import { Banner, Loading, Panel, Tag, fmt, useAsync } from "../components";

type Layer = "original" | "heatmap" | "overlay";
const MAX_MB = 10;

function Result({ p, url, layer, opacity }: { p: Prediction; url: string; layer: Layer; opacity: number }) {
  const src = layer === "original" ? url : `data:image/png;base64,${layer === "heatmap" ? p.heatmap_png_base64 : p.overlay_png_base64}`;
  return (
    <div className="space-y-3">
      <div className="relative">
        <img src={url} alt="Uploaded" className="w-full rounded border" />
        {layer !== "original" && <img src={src} alt={`${layer} (${p.model_name})`} style={{ opacity }} className="absolute inset-0 h-full w-full rounded" />}
      </div>
      <div className="flex flex-wrap items-center gap-2 text-sm">
        <Tag tone={p.predicted_status === "normal" ? "ok" : "bad"}>{p.predicted_status}</Tag>
        <span>score <b>{fmt(p.image_level_anomaly_score, 2)}</b> (threshold {fmt(p.threshold_used, 2)})</span>
        <span className="text-slate-500">evidence: {p.confidence_or_evidence}</span>
      </div>
      <p className="text-xs text-slate-500">{p.score_semantics}. Appearance z {fmt(p.appearance_z, 2)} · relation z {fmt(p.relation_z, 2)} · {p.inference_latency_ms.toFixed(0)} ms on {p.device} · {p.checkpoint_id}</p>
      <p className="rounded bg-slate-50 p-2 text-sm">{p.explanation.text}</p>
      {p.warnings.map((w) => <Banner key={w}>{w}</Banner>)}
      {p.relation_violations.length > 0 && (
        <div className="overflow-x-auto"><table className="w-full text-left text-xs"><thead><tr className="text-slate-500"><th className="pr-3">Type</th><th className="pr-3">Region type #</th><th className="pr-3">Observed / expected</th><th className="pr-3">z</th><th>Evidence</th></tr></thead>
          <tbody>{p.relation_violations.map((v, i) => <tr key={i} className="border-t"><td className="pr-3">{v.type}</td><td className="pr-3">{v.proto}</td><td className="pr-3">{v.observed !== undefined ? `${v.observed} / ${fmt(v.expected, 1)}` : "–"}</td><td className="pr-3">{fmt(v.z, 1)}</td><td>{v.evidence}</td></tr>)}</tbody></table></div>)}
      {p.regions.length > 0 && <details className="text-xs"><summary className="cursor-pointer text-slate-600">{p.regions.length} regions considered</summary>
        <div className="overflow-x-auto"><table className="mt-1 w-full text-left"><thead><tr className="text-slate-500"><th>Type #</th><th>cx</th><th>cy</th><th>area</th><th>relation z</th><th>appearance z</th></tr></thead>
          <tbody>{p.regions.map((r, i) => <tr key={i} className="border-t"><td>{r.proto}</td><td>{fmt(r.cx, 2)}</td><td>{fmt(r.cy, 2)}</td><td>{fmt(r.area, 3)}</td><td>{fmt(r.relation_z, 1)}</td><td>{fmt(r.appearance_z, 1)}</td></tr>)}</tbody></table></div></details>}
    </div>
  );
}

export default function Inspect() {
  const models = useAsync(api.models);
  const [modelId, setModelId] = useState("");
  const [file, setFile] = useState<File | null>(null);
  const [url, setUrl] = useState("");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [single, setSingle] = useState<Prediction | null>(null);
  const [cmp, setCmp] = useState<Compare | null>(null);
  const [layer, setLayer] = useState<Layer>("overlay");
  const [opacity, setOpacity] = useState(0.8);

  const groups = useMemo(() => {
    const g: Record<string, ModelInfo[]> = {};
    (models.data ?? []).forEach((m) => { (g[`${m.run}/${m.category}`] ??= []).push(m); });
    return g;
  }, [models.data]);
  useEffect(() => { if (!modelId && models.data?.length) setModelId(models.data[0].model_id); }, [models.data, modelId]);
  useEffect(() => () => { if (url) URL.revokeObjectURL(url); }, [url]);

  const pick = (f: File | undefined) => {
    setSingle(null); setCmp(null); setError(null);
    if (!f) return;
    if (!["image/png", "image/jpeg"].includes(f.type)) return setError("Please choose a JPG or PNG image.");
    if (f.size > MAX_MB * 2 ** 20) return setError(`File exceeds ${MAX_MB} MB.`);
    setFile(f); setUrl(URL.createObjectURL(f));
  };
  const run = async (mode: "single" | "compare") => {
    if (!file) return;
    setBusy(true); setError(null); setSingle(null); setCmp(null);
    try {
      if (mode === "single") setSingle(await api.predict(file, modelId));
      else { const [run, category] = modelId.split("/"); setCmp(await api.compare(file, run, category)); }
    } catch (e) { setError(String((e as Error).message)); } finally { setBusy(false); }
  };
  const sel = models.data?.find((m) => m.model_id === modelId);
  const canCompare = sel && (groups[`${sel.run}/${sel.category}`] ?? []).length === 2;

  return (
    <div className="space-y-4">
      <h1 className="text-xl font-semibold text-ink">Image inspection</h1>
      <Panel>
        {models.loading ? <Loading /> : models.error ? <Banner tone="error">{models.error}</Banner> : !models.data?.length ? (
          <Banner>No model checkpoints are available, so inference is disabled. Train a model first (see README).</Banner>
        ) : (
          <div className="grid gap-3 md:grid-cols-[1fr_1fr_auto] md:items-end">
            <label className="text-sm">Image (JPG/PNG, ≤ {MAX_MB} MB)<input type="file" accept="image/png,image/jpeg" onChange={(e) => pick(e.target.files?.[0])} className="mt-1 block w-full text-sm" /></label>
            <label className="text-sm">Model<select value={modelId} onChange={(e) => setModelId(e.target.value)} className="mt-1 block w-full rounded border px-2 py-1.5">
              {Object.entries(groups).map(([g, ms]) => <optgroup key={g} label={g}>{ms.map((m) => <option key={m.model_id} value={m.model_id}>{m.kind}{m.smoke_only ? " (smoke)" : ""}</option>)}</optgroup>)}</select></label>
            <div className="flex gap-2">
              <button disabled={!file || busy} onClick={() => run("single")} className="rounded bg-ink px-3 py-1.5 text-sm text-white disabled:opacity-40">Analyze</button>
              <button disabled={!file || busy || !canCompare} onClick={() => run("compare")} title={canCompare ? "" : "Needs both baseline and relation checkpoints"} className="rounded border border-ink px-3 py-1.5 text-sm text-ink disabled:opacity-40">Compare both</button>
            </div>
          </div>)}
        {sel?.smoke_only && <div className="mt-3"><Banner>This checkpoint is a smoke test (random-init backbone and/or synthetic data). Scores are not meaningful benchmark results.</Banner></div>}
      </Panel>
      {error && <Banner tone="error">{error}</Banner>}
      {busy && <Loading />}
      {file && !single && !cmp && !busy && <Panel title="Preview"><img src={url} alt="Preview" className="max-h-80 rounded border" /></Panel>}
      {(single || cmp) && (
        <Panel>
          <div className="mb-3 flex flex-wrap items-center gap-4 text-sm">
            <div role="group" aria-label="Layer" className="flex overflow-hidden rounded border">
              {(["original", "heatmap", "overlay"] as Layer[]).map((l) => <button key={l} onClick={() => setLayer(l)} aria-pressed={layer === l} className={`px-3 py-1 ${layer === l ? "bg-ink text-white" : "bg-white"}`}>{l}</button>)}</div>
            <label className="flex items-center gap-2">Layer opacity<input type="range" min={0.1} max={1} step={0.05} value={opacity} onChange={(e) => setOpacity(+e.target.value)} /></label>
            <span className="text-xs text-slate-500">Display controls only; they never change the stored score or status.</span>
          </div>
          {single && <Result p={single} url={url} layer={layer} opacity={opacity} />}
          {cmp && <div className="grid gap-6 md:grid-cols-2">{(["baseline", "relation"] as const).map((k) => <div key={k}><h3 className="mb-2 font-medium capitalize">{k === "baseline" ? "Appearance baseline" : "Relation-aware"}</h3><Result p={cmp[k]} url={url} layer={layer} opacity={opacity} /></div>)}</div>}
        </Panel>)}
    </div>
  );
}
