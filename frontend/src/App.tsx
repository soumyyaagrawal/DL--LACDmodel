import { NavLink, Route, Routes } from "react-router-dom";
import Dashboard from "./pages/Dashboard";
import Inspect from "./pages/Inspect";
import Evaluation from "./pages/Evaluation";
import Dataset from "./pages/Dataset";
import Workspace from "./pages/Workspace";

const links = [["/", "Dashboard"], ["/inspect", "Inspect"], ["/evaluation", "Evaluation"], ["/dataset", "Dataset"], ["/workspace", "Local dataset"]];

export default function App() {
  return (
    <div className="min-h-screen">
      <header className="border-b bg-white">
        <div className="mx-auto flex max-w-6xl flex-wrap items-center gap-x-6 gap-y-2 px-4 py-3">
          <span className="font-semibold text-ink">LACD <span className="font-normal text-slate-500">· Logical Assembly Error Detection</span></span>
          <nav className="flex flex-wrap gap-1 text-sm" aria-label="Main">
            {links.map(([to, label]) => (
              <NavLink key={to} to={to} end={to === "/"} className={({ isActive }) => `rounded px-3 py-1.5 ${isActive ? "bg-ink text-white" : "text-slate-700 hover:bg-slate-100"}`}>{label}</NavLink>
            ))}
          </nav>
        </div>
      </header>
      <main className="mx-auto max-w-6xl px-4 py-6">
        <Routes>
          <Route path="/" element={<Dashboard />} />
          <Route path="/inspect" element={<Inspect />} />
          <Route path="/evaluation" element={<Evaluation />} />
          <Route path="/dataset" element={<Dataset />} />
          <Route path="/workspace" element={<Workspace />} />
        </Routes>
      </main>
    </div>
  );
}
