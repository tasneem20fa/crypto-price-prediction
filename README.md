README.md


**Project summary**  
End-to-end cryptocurrency price prediction system (BTC). Trained a supervised model on historical data, exposed prediction endpoints via Flask, and built a React dashboard to visualize predictions.

---

## What’s included
- `backend/` — Flask API (`app.py`) serving:
  - `GET /` — health
  - `GET /predict_latest` — predict next close from latest data
  - `GET /predict_horizon?n=7` — iterative forecast for n days
- `model/` — saved model and scaler (`crypto_model_enhanced.pkl`, optionally `scaler_enhanced.pkl`)
- `data/` — prepared files (`features_enhanced.csv`, `BTC_daily_2017.csv`)
- `frontend/` — React app (Vite) that shows latest prediction and horizon chart
- `requirements.txt` — Python dependencies
- `README.md` — this file

---

## Quick start (how to run locally)

### 1. Backend
```bash
# create & activate venv (Windows PowerShell)
python -m venv venv
.\venv\Scripts\Activate.ps1

# install
pip install -r requirements.txt

# start API
python backend/app.py
