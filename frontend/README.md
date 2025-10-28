# Crypto Price Prediction — (Your Name / Team)

**Project Summary:**  
End-to-end cryptocurrency price prediction system using Bitcoin data.  
Built a supervised ML model, served predictions with a Flask API, and visualized results on a React dashboard.

---

## 📁 Project Structure

- `backend/` — Flask API serving:
  - `GET /` — health check  
  - `GET /predict_latest` — latest BTC close prediction  
  - `GET /predict_horizon?n=7` — 7-day forecast  

- `model/` — saved ML model, scaler, and metrics  
- `data/` — processed CSV data (`features_enhanced.csv`)  
- `frontend/` — React (Vite) dashboard  
- `requirements.txt` — Python dependencies  

---

## ⚙️ How to Run (Locally)

### Backend
```bash
# activate venv
python -m venv venv
.\venv\Scripts\Activate.ps1

# install dependencies
pip install -r requirements.txt

# start backend
python backend/app.py
