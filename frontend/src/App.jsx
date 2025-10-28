// frontend/src/App.jsx
import React, { useEffect, useRef, useState } from "react";
import { getLatestPrediction, getFuturePredictions } from "./api";

export default function App() {
  const [latest, setLatest] = useState(null);
  const [loadingLatest, setLoadingLatest] = useState(false);
  const [horizon, setHorizon] = useState(7);
  const [predictions, setPredictions] = useState([]);
  const [loadingHorizon, setLoadingHorizon] = useState(false);
  const [error, setError] = useState(null);
  const [showFeatures, setShowFeatures] = useState(false);
  const canvasRef = useRef(null);

  // fetch latest on mount
  useEffect(() => {
    fetchLatest();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  useEffect(() => {
    // draw chart whenever predictions change
    if (predictions && predictions.length) {
      drawChart(predictions);
    } else {
      clearCanvas();
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [predictions]);

  async function fetchLatest() {
    setLoadingLatest(true);
    setError(null);
    try {
      const res = await getLatestPrediction();
      if (res.error) throw new Error(res.error);
      setLatest(res);
    } catch (err) {
      setError(err.message || "Failed to fetch latest prediction");
    } finally {
      setLoadingLatest(false);
    }
  }

  async function fetchHorizon() {
    setLoadingHorizon(true);
    setError(null);
    try {
      const res = await getFuturePredictions(horizon);
      if (res.error) throw new Error(res.error);
      if (!res.predictions) throw new Error("No predictions returned");
      setPredictions(res.predictions);
    } catch (err) {
      setError(err.message || "Failed to fetch horizon predictions");
      setPredictions([]);
    } finally {
      setLoadingHorizon(false);
    }
  }

  function clearCanvas() {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    ctx.clearRect(0, 0, canvas.width, canvas.height);
  }

  function drawChart(data) {
    // data: [{date: "YYYY-MM-DD", predicted_close: 1234}, ...]
    const canvas = canvasRef.current;
    if (!canvas || !data || !data.length) return;
    const ctx = canvas.getContext("2d");

    // size & pixel ratio
    const DPR = window.devicePixelRatio || 1;
    const W = 800;
    const H = 300;
    canvas.width = W * DPR;
    canvas.height = H * DPR;
    canvas.style.width = `${W}px`;
    canvas.style.height = `${H}px`;
    ctx.scale(DPR, DPR);

    // clear
    ctx.clearRect(0, 0, W, H);

    // margins
    const margin = { top: 20, right: 20, bottom: 40, left: 60 };
    const plotW = W - margin.left - margin.right;
    const plotH = H - margin.top - margin.bottom;

    // values
    const values = data.map((d) => Number(d.predicted_close));
    const dates = data.map((d) => d.date);

    // scales
    const minVal = Math.min(...values);
    const maxVal = Math.max(...values);
    const pad = (maxVal - minVal) * 0.12 || maxVal * 0.05;
    const yMin = minVal - pad;
    const yMax = maxVal + pad;

    const xToPixel = (i) =>
      margin.left + (i / Math.max(1, values.length - 1)) * plotW;
    const yToPixel = (v) =>
      margin.top + ((yMax - v) / (yMax - yMin)) * plotH;

    // gridlines and y axis ticks
    ctx.font = "12px sans-serif";
    ctx.fillStyle = "#222";
    ctx.strokeStyle = "#e6e6e6";
    ctx.lineWidth = 1;

    const ticks = 5;
    for (let t = 0; t <= ticks; t++) {
      const yv = yMin + (t / ticks) * (yMax - yMin);
      const y = yToPixel(yv);
      // grid
      ctx.beginPath();
      ctx.moveTo(margin.left, y);
      ctx.lineTo(margin.left + plotW, y);
      ctx.stroke();
      // label
      ctx.fillStyle = "#333";
      ctx.textAlign = "right";
      ctx.fillText(yv.toFixed(0), margin.left - 8, y + 4);
    }

    // x labels
    ctx.textAlign = "center";
    ctx.fillStyle = "#333";
    for (let i = 0; i < dates.length; i++) {
      const x = xToPixel(i);
      const label = dates[i].slice(5); // MM-DD
      ctx.fillText(label, x, margin.top + plotH + 18);
    }

    // draw line
    ctx.beginPath();
    ctx.lineWidth = 2.5;
    ctx.strokeStyle = "#0f62fe";
    for (let i = 0; i < values.length; i++) {
      const x = xToPixel(i);
      const y = yToPixel(values[i]);
      if (i === 0) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
    }
    ctx.stroke();

    // draw area under curve
    ctx.lineTo(margin.left + plotW, margin.top + plotH);
    ctx.lineTo(margin.left, margin.top + plotH);
    ctx.closePath();
    ctx.fillStyle = "rgba(15,98,254,0.08)";
    ctx.fill();

    // draw points
    ctx.fillStyle = "#ffffff";
    ctx.strokeStyle = "#0850b8";
    for (let i = 0; i < values.length; i++) {
      const x = xToPixel(i);
      const y = yToPixel(values[i]);
      ctx.beginPath();
      ctx.arc(x, y, 4, 0, Math.PI * 2);
      ctx.fill();
      ctx.stroke();
    }

    // title
    ctx.fillStyle = "#111";
    ctx.textAlign = "left";
    ctx.font = "14px sans-serif";
    ctx.fillText(`Next ${values.length} days forecast`, margin.left, 14);
  }

  return (
    <div style={styles.container}>
      <h1 style={styles.h1}>Crypto Price Dashboard</h1>

      <section style={styles.card}>
        <h2 style={styles.h2}>Latest prediction</h2>
        {loadingLatest ? (
          <p>Loading latest prediction…</p>
        ) : error ? (
          <p style={{ color: "red" }}>{error}</p>
        ) : latest ? (
          <div>
            <p>
              <strong>Date used:</strong> {latest.date_used ?? "—"}
            </p>
            <p>
              <strong>Predicted next close:</strong>{" "}
              <span style={styles.price}>
                {latest.predicted_next_close?.toFixed(2) ?? "—"}
              </span>
            </p>

            <button
              style={styles.button}
              onClick={() => setShowFeatures((s) => !s)}
            >
              {showFeatures ? "Hide features" : "Show features"}
            </button>

            {showFeatures && latest.feature_values && (
              <pre style={styles.pre}>
                {JSON.stringify(latest.feature_values, null, 2)}
              </pre>
            )}
          </div>
        ) : (
          <p>No data yet</p>
        )}
      </section>

      <section style={styles.card}>
        <h2 style={styles.h2}>Forecast horizon</h2>

        <div style={{ display: "flex", gap: 8, alignItems: "center" }}>
          <label>
            Days:
            <select
              value={horizon}
              onChange={(e) => setHorizon(Number(e.target.value))}
              style={{ marginLeft: 8 }}
            >
              {[3, 5, 7, 14, 21, 30].map((d) => (
                <option key={d} value={d}>
                  {d}
                </option>
              ))}
            </select>
          </label>

          <button style={styles.button} onClick={fetchHorizon} disabled={loadingHorizon}>
            {loadingHorizon ? "Predicting…" : "Get forecast"}
          </button>
        </div>

        <div style={{ marginTop: 16 }}>
          <canvas ref={canvasRef} width={800} height={300} style={{ maxWidth: "100%" }} />
        </div>

        <div style={{ marginTop: 12 }}>
          {predictions && predictions.length > 0 && (
            <table style={styles.table}>
              <thead>
                <tr>
                  <th>Date</th>
                  <th>Predicted Close</th>
                </tr>
              </thead>
              <tbody>
                {predictions.map((p) => (
                  <tr key={p.date}>
                    <td style={{ textAlign: "center" }}>{p.date}</td>
                    <td style={{ textAlign: "center" }}>{p.predicted_close}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          )}
        </div>
      </section>

      <footer style={styles.footer}>
        Backend: <code>http://127.0.0.1:5000</code> — Frontend (Vite): <code>http://localhost:5173</code>
      </footer>
    </div>
  );
}

// ---- styles ----
const styles = {
  container: {
    fontFamily: "Inter, Roboto, system-ui, -apple-system, 'Segoe UI', Arial",
    padding: 20,
    maxWidth: 960,
    margin: "0 auto",
  },
  h1: { margin: "8px 0 18px 0" },
  card: {
    background: "#fff",
    borderRadius: 8,
    boxShadow: "0 6px 20px rgba(0,0,0,0.06)",
    padding: 16,
    marginBottom: 18,
  },
  h2: { margin: "0 0 8px 0" },
  button: {
    padding: "8px 12px",
    borderRadius: 6,
    border: "1px solid #0850b8",
    background: "#0f62fe",
    color: "white",
    cursor: "pointer",
  },
  price: { fontSize: 20, color: "#0850b8" },
  pre: {
    background: "#f7f7fb",
    padding: 10,
    borderRadius: 6,
    marginTop: 10,
    maxHeight: 240,
    overflow: "auto",
  },
  table: {
    width: "100%",
    borderCollapse: "collapse",
    marginTop: 8,
  },
  footer: { marginTop: 18, color: "#666", fontSize: 13 },
};
