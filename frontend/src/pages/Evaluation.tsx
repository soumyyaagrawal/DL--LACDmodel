import { api, EvalResult } from "../api";
import { Banner, Loading, Panel, fmt, useAsync } from "../components";

function Roc({ r }: { r: EvalResult }) {
  if (!r.roc || !r.roc.fpr.length) return <p className="text-xs text-slate-500">ROC unavailable.</p>;
  const pts = r.roc.fpr.map((x, i) => `${(x * 100).toFixed(1)},${(100 - r.roc!.tpr[i] * 100).toFixed(1)}`).join(" ");
  return <svg viewBox="-8 -4 116 116" className="h-40 w-40" role="img" aria-label={`ROC curve ${r.model}`}><rect x="0" y="0" width="100" height="100" fill="none" stroke="#cbd5e1" /><line x1="0" y1="100" x2="100" y2="0" stroke="#cbd5e1" strokeDasharray="3" /><polyline points={pts} fill="none" stroke="#0f766e" strokeWidth="1.5" /><text x="50" y="112" fontSize="6" textAnchor="middle">FPR</text></svg>;
}
function Cm({ m }: { m: number[][] }) {
  return <table className="text-center text-xs"><tbody>{[["TN", "FP"], ["FN", "TP"]].map((row, i) => <tr key={i}>{row.map((l, j) => <td key={l} className="border px-3 py-2"><div className="text-slate-500">{l}</div><b>{m[i][j]}</b></td>)}</tr>)}</tbody></table>;
}

export default function Evaluation() {
  const ev = useAsync(api.evalSummary);
  if (ev.loading) return <Loading />;
  if (ev.error) return <Banner tone="error">{ev.error}</Banner>;
  if (!ev.data?.available) return <div className="space-y-3"><h1 className="text-xl font-semibold text-ink">Model evaluation</h1><Banner tone="info">Pending: {ev.data?.message}</Banner></div>;
  const rs = ev.data.results ?? [];
  const cats = [...new Set(rs.map((r) => `${r.experiment}/${r.category}`))];
  return (
    <div className="space-y-5">
      <h1 className="text-xl font-semibold text-ink">Model evaluation</h1>
      {cats.map((c) => {
        const rows = rs.filter((r) => `${r.experiment}/${r.category}` === c);
        const flagged = rows.some((r) => r.synthetic_data || r.weights_source === "random_init");
        return (
          <Panel key={c} title={c}>
            {flagged && <div className="mb-3"><Banner>These numbers come from SYNTHETIC data and/or a random-init backbone (pipeline smoke test). They are not benchmark results and do not show that either method is better.</Banner></div>}
            <div className="overflow-x-auto"><table className="w-full text-left text-sm"><thead><tr className="text-slate-500"><th className="pr-4">Model</th><th className="pr-4">Subset</th><th className="pr-4">n</th><th className="pr-4">AUROC</th><th className="pr-4">Prec.</th><th className="pr-4">Recall</th><th className="pr-4">F1</th><th className="pr-4">Pixel AUROC</th><th className="pr-4">IoU</th><th>Dice</th></tr></thead>
              <tbody>{rows.flatMap((r) => [["all", r.overall, r.pixel.overall] as const, ...Object.entries(r.by_type).map(([t, m]) => [t, m, r.pixel[t]] as const)].map(([name, m, p]) => (
                <tr key={r.model + name} className="border-t"><td className="pr-4">{r.model}</td><td className="pr-4">{name === "all" ? "all anomalies" : name}</td><td className="pr-4">{m.n}</td><td className="pr-4">{fmt(m.auroc)}</td><td className="pr-4">{fmt(m.precision)}</td><td className="pr-4">{fmt(m.recall)}</td><td className="pr-4">{fmt(m.f1)}</td><td className="pr-4">{fmt(p?.pixel_auroc)}</td><td className="pr-4">{fmt(p?.iou_mean_anomalous)}</td><td>{fmt(p?.dice_mean_anomalous)}</td></tr>)))}</tbody></table></div>
            <div className="mt-4 grid gap-6 md:grid-cols-2">{rows.map((r) => (
              <div key={r.model} className="space-y-2 text-sm"><h3 className="font-medium">{r.model}</h3><div className="flex flex-wrap items-center gap-4"><Roc r={r} /><Cm m={r.overall.confusion_matrix} /></div>
                <p className="text-xs text-slate-500">Threshold {fmt(r.overall.threshold, 2)} — {r.threshold_procedure}. Latency {fmt(r.latency?.mean_ms, 1)} ms on {r.latency?.device}; peak GPU memory: {r.latency?.peak_gpu_mem_mb ?? r.latency?.peak_gpu_mem_note ?? "n/a"}. Checkpoint {r.checkpoint_id}.</p></div>))}</div>
          </Panel>);
      })}
    </div>
  );
}
