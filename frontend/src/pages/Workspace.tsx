import { api } from "../api";
import { Banner, Loading, Panel, useAsync } from "../components";

export default function Workspace() {
  const ds = useAsync(api.datasetSummary);
  const local = ds.data?.local_tabletop;
  return (
    <div className="space-y-4">
      <h1 className="text-xl font-semibold text-ink">Local tabletop dataset</h1>
      <Panel title="Status">{ds.loading ? <Loading /> : local?.available ? <p className="text-sm">{local.total_images} images found on disk.</p> : <Banner tone="info">No local dataset found at the configured location (LACD_LOCAL_DATA_ROOT). Nothing has been collected or validated yet.</Banner>}</Panel>
      <Panel title="Layout">
        <pre className="overflow-x-auto rounded bg-slate-50 p-3 text-xs">{`<root>/<assembly_type>/
  train/good/*.png
  validation/good/*.png
  test/good/*.png
  test/<anomaly_type>/*.png        (e.g. missing_component, extra_component, misplaced_component, structural_anomalies)
  ground_truth/<anomaly_type>/<image_stem>/000.png   (optional binary masks)
  annotations.json                 (optional: {"test/<type>/<file>": {"subtype": "missing"}})
  assembly_spec.json               (optional: expected components and positions)`}</pre>
        <p className="mt-2 text-sm text-slate-600">Uploading and batch validation through this page are not implemented yet; copy files into the folder above and run <code>scripts/inspect_dataset.py</code>. Uploaded files are never treated as trained or validated samples automatically.</p>
      </Panel>
    </div>
  );
}
