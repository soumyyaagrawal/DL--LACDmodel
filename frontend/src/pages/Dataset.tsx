import { api } from "../api";
import { Banner, Empty, Loading, Panel, useAsync } from "../components";

export default function Dataset() {
  const ds = useAsync(api.datasetSummary);
  if (ds.loading) return <Loading />;
  if (ds.error) return <Banner tone="error">{ds.error}</Banner>;
  return (
    <div className="space-y-4">
      <h1 className="text-xl font-semibold text-ink">Dataset explorer</h1>
      {Object.entries(ds.data ?? {}).map(([name, v]) => (
        <Panel key={name} title={name}>
          {!v.available ? <Empty>{v.message} Download MVTec LOCO AD from the official page and set LACD_DATA_ROOT (see DATASET_GUIDE.md).</Empty> : (
            <div className="overflow-x-auto"><table className="w-full text-left text-sm"><thead><tr className="text-slate-500"><th className="pr-4">Category</th><th className="pr-4">Split / type</th><th>Images</th></tr></thead>
              <tbody>{Object.entries(v.categories ?? {}).flatMap(([c, counts]) => Object.entries(counts).map(([k, n]) => <tr key={c + k} className="border-t"><td className="pr-4">{c}</td><td className="pr-4">{k}</td><td>{n}</td></tr>))}</tbody></table></div>)}
        </Panel>))}
      <p className="text-xs text-slate-500">Counts are read from disk. The proposal cites 3,644 LOCO images; compare with the total shown for your download. Image/annotation browsing is not implemented yet.</p>
    </div>
  );
}
