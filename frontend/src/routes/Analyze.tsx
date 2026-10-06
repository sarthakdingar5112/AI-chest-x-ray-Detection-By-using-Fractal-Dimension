import { useCallback, useEffect, useState } from "react";
import { AnimatePresence, motion } from "framer-motion";
import UploadDropzone from "../components/UploadDropzone";
import LoadingScanner from "../components/LoadingScanner";
import AnalysisView from "../components/AnalysisView";
import { analyzeImage } from "../lib/api";
import type { AnalysisResponse } from "../types/analysis";

type State =
  | { phase: "idle" }
  | { phase: "loading"; preview: string }
  | { phase: "done"; result: AnalysisResponse }
  | { phase: "error"; message: string };

export default function Analyze() {
  const [state, setState] = useState<State>({ phase: "idle" });

  const onFile = useCallback(async (file: File) => {
    const preview = URL.createObjectURL(file);
    setState({ phase: "loading", preview });
    try {
      const result = await analyzeImage(file);
      setState({ phase: "done", result });
    } catch (e) {
      setState({ phase: "error", message: e instanceof Error ? e.message : "Analysis failed." });
    }
  }, []);

  useEffect(() => {
    return () => {
      if (state.phase === "loading") URL.revokeObjectURL(state.preview);
    };
  }, [state]);

  return (
    <div className="container" style={{ padding: "24px 0 40px" }}>
      <AnimatePresence mode="wait">
        {state.phase === "idle" && (
          <motion.div key="idle" initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }}>
            <div style={{ maxWidth: 620, margin: "40px auto 28px", textAlign: "center" }}>
              <h1 style={{ fontSize: 34 }}>Analyze a chest X-ray</h1>
              <p className="dim" style={{ marginTop: 10, lineHeight: 1.6 }}>
                Upload an image to run multi-disease classification alongside a full fractal
                dimension analysis — including the local FD complexity heatmap.
              </p>
            </div>
            <div style={{ maxWidth: 620, margin: "0 auto" }}>
              <UploadDropzone onFile={onFile} />
            </div>
          </motion.div>
        )}

        {state.phase === "loading" && (
          <motion.div key="loading" initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }} style={{ maxWidth: 460, margin: "40px auto" }}>
            <LoadingScanner src={state.preview} />
            <p className="muted" style={{ textAlign: "center", marginTop: 18, fontSize: 14 }}>
              Computing box-counting dimension, lacunarity and the multifractal spectrum…
            </p>
          </motion.div>
        )}

        {state.phase === "done" && (
          <motion.div key="done" initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }}>
            <AnalysisView result={state.result} onReset={() => setState({ phase: "idle" })} />
          </motion.div>
        )}

        {state.phase === "error" && (() => {
          const notXray = /x-ray|lung|color image/i.test(state.message);
          return (
            <motion.div key="error" initial={{ opacity: 0 }} animate={{ opacity: 1 }} style={{ maxWidth: 520, margin: "60px auto", textAlign: "center" }}>
              <div className="card" style={{ padding: 30 }}>
                <div style={{ fontSize: 40, marginBottom: 10 }}>{notXray ? "🫁" : "⚠️"}</div>
                <h3>{notXray ? "That doesn't look like a chest X-ray" : "Analysis failed"}</h3>
                <p className="muted" style={{ marginTop: 8, fontSize: 14, lineHeight: 1.6 }}>{state.message}</p>
                <button className="btn" style={{ marginTop: 18 }} onClick={() => setState({ phase: "idle" })}>
                  {notXray ? "Upload a chest X-ray" : "Try again"}
                </button>
              </div>
            </motion.div>
          );
        })()}
      </AnimatePresence>
    </div>
  );
}
