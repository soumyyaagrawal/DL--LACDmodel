import { ReactNode, useEffect, useState } from "react";

export function useAsync<T>(fn: () => Promise<T>, deps: unknown[] = []) {
  const [data, setData] = useState<T | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);
  useEffect(() => {
    let live = true;
    setLoading(true);
    fn().then((d) => live && (setData(d), setError(null))).catch((e) => live && setError(String(e.message ?? e))).finally(() => live && setLoading(false));
    return () => { live = false; };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, deps);
  return { data, error, loading };
}

export const Panel = ({ title, children }: { title?: string; children: ReactNode }) => (
  <section className="rounded-lg border bg-white p-4">{title && <h2 className="mb-3 text-sm font-semibold uppercase tracking-wide text-slate-500">{title}</h2>}{children}</section>
);
export const Banner = ({ tone = "warn", children }: { tone?: "warn" | "error" | "info"; children: ReactNode }) => {
  const c = { warn: "border-amber-300 bg-amber-50 text-amber-900", error: "border-red-300 bg-red-50 text-red-900", info: "border-sky-300 bg-sky-50 text-sky-900" }[tone];
  return <div role={tone === "error" ? "alert" : "status"} className={`rounded border px-3 py-2 text-sm ${c}`}>{children}</div>;
};
export const Loading = () => <p className="text-sm text-slate-500" aria-live="polite">Loading…</p>;
export const Empty = ({ children }: { children: ReactNode }) => <p className="rounded border border-dashed p-6 text-center text-sm text-slate-500">{children}</p>;
export const fmt = (v: number | null | undefined, d = 3) => (v === null || v === undefined ? "n/a" : v.toFixed(d));
export const Tag = ({ children, tone }: { children: ReactNode; tone: "ok" | "bad" | "muted" }) => (
  <span className={`rounded px-2 py-0.5 text-xs font-medium ${tone === "ok" ? "bg-emerald-100 text-emerald-800" : tone === "bad" ? "bg-red-100 text-red-800" : "bg-slate-100 text-slate-700"}`}>{children}</span>
);
