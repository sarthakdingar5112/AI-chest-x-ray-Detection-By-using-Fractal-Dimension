import { Link, Route, Routes, useLocation } from "react-router-dom";
import { AnimatePresence, motion } from "framer-motion";
import Home from "./routes/Home";
import Analyze from "./routes/Analyze";
import Methodology from "./routes/Methodology";
import DatasetStudy from "./routes/DatasetStudy";

function NavBar() {
  const loc = useLocation();
  const links = [
    { to: "/", label: "Home" },
    { to: "/analyze", label: "Analyze" },
    { to: "/methodology", label: "Methodology" },
    { to: "/dataset-study", label: "Dataset Study" },
  ];
  return (
    <header style={{ position: "sticky", top: 0, zIndex: 20 }}>
      <div
        className="container"
        style={{
          display: "flex",
          alignItems: "center",
          justifyContent: "space-between",
          padding: "18px 0",
        }}
      >
        <Link to="/" style={{ display: "flex", alignItems: "center", gap: 10 }}>
          <LogoMark />
          <strong style={{ fontSize: 18, letterSpacing: "-0.02em" }}>
            Fractal<span className="accent-text">Scan</span>
          </strong>
        </Link>
        <nav style={{ display: "flex", gap: 6 }}>
          {links.map((l) => {
            const active = loc.pathname === l.to;
            return (
              <Link
                key={l.to}
                to={l.to}
                className="badge"
                style={{
                  borderColor: active ? "var(--border-strong)" : "transparent",
                  color: active ? "var(--text-0)" : "var(--text-2)",
                  background: active ? "rgba(53,224,212,0.08)" : "transparent",
                }}
              >
                {l.label}
              </Link>
            );
          })}
        </nav>
      </div>
    </header>
  );
}

function LogoMark() {
  return (
    <svg width="26" height="26" viewBox="0 0 26 26" fill="none">
      <rect x="1" y="1" width="24" height="24" rx="7" stroke="url(#g)" strokeWidth="1.5" />
      <path d="M6 18 L10 8 L13 15 L16 6 L20 18" stroke="url(#g)" strokeWidth="1.8" fill="none" strokeLinejoin="round" strokeLinecap="round" />
      <defs>
        <linearGradient id="g" x1="0" y1="0" x2="26" y2="26">
          <stop stopColor="#35e0d4" />
          <stop offset="1" stopColor="#4d8dff" />
        </linearGradient>
      </defs>
    </svg>
  );
}

export default function App() {
  const loc = useLocation();
  return (
    <div className="grid-bg" style={{ position: "relative", minHeight: "100%", zIndex: 1 }}>
      <NavBar />
      <AnimatePresence mode="wait">
        <motion.main
          key={loc.pathname}
          initial={{ opacity: 0, y: 12 }}
          animate={{ opacity: 1, y: 0 }}
          exit={{ opacity: 0, y: -8 }}
          transition={{ duration: 0.35, ease: [0.22, 1, 0.36, 1] }}
        >
          <Routes location={loc}>
            <Route path="/" element={<Home />} />
            <Route path="/analyze" element={<Analyze />} />
            <Route path="/methodology" element={<Methodology />} />
            <Route path="/dataset-study" element={<DatasetStudy />} />
          </Routes>
        </motion.main>
      </AnimatePresence>
      <footer className="container" style={{ padding: "60px 0 40px", color: "var(--text-2)", fontSize: 13 }}>
        FractalScan · BE Final-Year Project · Fractal Dimension Analysis for X-ray abnormality detection.
        <br />
        Research/education use only — not a medical device.
      </footer>
    </div>
  );
}
