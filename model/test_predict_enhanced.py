# model/test_predict_enhanced.py
import joblib, os, numpy as np, pandas as pd

BASE = os.path.dirname(os.path.abspath(__file__))
M = joblib.load(os.path.join(BASE, "crypto_model_enhanced.pkl"))
model = M['model']; scaler = M['scaler']; features = M['features']

# Build a sample row by reading last row of features_enhanced.csv (simulate latest)
df = pd.read_csv(os.path.join(BASE, "..", "data", "features_enhanced.csv"), parse_dates=['date'])
row = df.tail(1)[features]
X = scaler.transform(row.values)
pred_next_close = model.predict(X)[0]
print("Predicted next-day close:", pred_next_close)
