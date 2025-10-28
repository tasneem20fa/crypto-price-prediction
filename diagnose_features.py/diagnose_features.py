import pandas as pd
import os
import json
from backend.app import compute_features_from_daily, _find_volume_column

BASE_DIR = os.path.dirname(__file__)
daily_path = os.path.join(BASE_DIR, "data", "BTC_daily_2017.csv")

print("📁 Daily CSV path:", daily_path)

# --- 1. Load the daily CSV
try:
    df = pd.read_csv(daily_path, nrows=10, parse_dates=['date'])
    full = pd.read_csv(daily_path, parse_dates=['date'])
    print(f"✅ Daily CSV loaded — shape: {full.shape}")
    print("Columns:", full.columns.tolist())
    print("\nPreview:")
    print(df.head())
except Exception as e:
    print("❌ Failed to load daily CSV:", e)
    raise SystemExit

# --- 2. Compute enhanced features
try:
    df_feat = compute_features_from_daily(full)
    print(f"\n✅ Computed features shape: {df_feat.shape}")
    print("Computed columns (first 20):", df_feat.columns.tolist()[:20])
    print("\nLast 5 rows of computed features:")
    print(df_feat.tail(5))
    print("\nNaN counts (top 20):")
    print(df_feat.isna().sum().sort_values(ascending=False).head(20))
except Exception as e:
    print("\n❌ Error during compute_features_from_daily():", e)
    raise SystemExit

# --- 3. Check what model expects
model_path = os.path.join(BASE_DIR, "model", "best_model.pkl")
if os.path.exists(model_path):
    import joblib
    model_package = joblib.load(model_path)
    features = model_package.get("features", [])
    print(f"\n📊 Model expects {len(features)} features.")
    missing = [f for f in features if f not in df_feat.columns]
    print("❗ Missing features in computed data:", missing)
else:
    print("\n⚠️ No model file found at:", model_path)

