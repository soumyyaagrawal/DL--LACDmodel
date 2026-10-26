import { api } from "../api";
import { Banner, Loading, Panel, useAsync, fmt } from "../components";

export default function Dashboard() {
  const health = useAsync(api.health);
  const models = useAsync(api.models);
  const ds = useAsync(api.datasetSummary);
  const ev = useAsync(api.evalSummary);
  return (
    <div className="space-y-4">
      <div>
        <h1 className="text-2xl font-semibold text-ink">Logical Assembly Error Detection Beyond Surface Defects</h1>
        <p className="mt-1 max-w-3xl text-sm text-slate-600">Compares an appearance-only anomaly detector with a relation-aware model that reasons about component counts, positions and pairwise arrangement. Deep Learning Lab, Ramdeobaba University, 2026–27.</p>
      </div>
      {health.error && <Banner tone="error">{health.error}</Banner>}
      <div className="grid gap-4 md:grid-cols-2">
        <Panel title="Backend">{health.loading ? <Loading /> : health.data && (
          <dl className="grid grid-cols-2 gap-1 text-sm"><dt className="text-slate-500">Status</dt><dd>{health.data.status}</dd><dt className="text-slate-500">PyTorch</dt><dd>{health.data.torch}</dd><dt className="text-slate-500">CUDA</dt><dd>{health.data.cuda_available ? "available" : "not available (CPU)"}</dd><dt className="text-slate-500">Checkpoints</dt><dd>{health.data.n_models}</dd></dl>)}</Panel>
        <Panel title="Datasets">{ds.loading ? <Loading /> : ds.data && Object.entries(ds.data).map(([k, v]) => (
          <p key={k} className="text-sm"><b>{k}</b>: {v.available ? `${v.total_images} images, categories: ${Object.keys(v.categories ?? {}).join(", ")}` : <span className="text-slate-500">not configured</span>}</p>))}</Panel>
        <Panel title="Registered models">{models.loading ? <Loading /> : models.data && models.data.length === 0 ? <p className="text-sm text-slate-500">No checkpoints found. Train with scripts/train_baseline.py and scripts/train_relation_model.py.</p> :
          <ul className="space-y-1 text-sm">{models.data?.map((m) => <li key={m.model_id}><code>{m.model_id}</code> · {m.checkpoint_id}{m.smoke_only && <span className="ml-2 rounded bg-amber-100 px-1.5 text-xs text-amber-900">smoke / random-init</span>}</li>)}</ul>}</Panel>
        <Panel title="Latest evaluation">{ev.loading ? <Loading /> : !ev.data?.available ? <p className="text-sm text-slate-500">{ev.data?.message ?? "Pending: no evaluation has been run."}</p> :
          <ul className="space-y-1 text-sm">{ev.data.results?.map((r) => <li key={r.experiment + r.category + r.model}>{r.category} · {r.model}: image AUROC {fmt(r.overall.auroc)} {r.synthetic_data && <span className="rounded bg-amber-100 px-1.5 text-xs text-amber-900">synthetic data, not a benchmark</span>}</li>)}</ul>}</Panel>
      </div>
    </div>
  );
}
