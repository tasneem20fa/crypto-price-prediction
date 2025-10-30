// frontend/src/api.js
const BASE_URL = import.meta.env.VITE_API_URL || "http://127.0.0.1:5000";

export async function getLatestPrediction() {
  const res = await fetch(`${BASE_URL}/predict_latest`);
  return await res.json();
}

export async function getFuturePredictions(days = 7) {
  const res = await fetch(`${BASE_URL}/predict_horizon?n=${days}`);
  return await res.json();
}

