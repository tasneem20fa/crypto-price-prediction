# model/train_model.py
import os
import sys
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.preprocessing import StandardScaler
import joblib
import json

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(BASE_DIR, "..", "data", "features_enhanced.csv")
MODEL_OUT = os.path.join(BASE_DIR, "crypto_model_enhanced.pkl")
METRICS_OUT = os.path.join(BASE_DIR, "metrics_enhanced.json")
IMPORTANCE_OUT = os.path.join(BASE_DIR, "feature_importances.csv")

def train_and_save():
    if not os.path.exists(DATA):
        print("ERROR: features file not found. Run prepare_data.py first.")
        sys.exit(1)

    df = pd.read_csv(DATA, parse_dates=['date'])
    print("Loaded rows:", len(df))
    # define target and features
    # target is next day's close — optionally create a shifted target
    df['target'] = df['close'].shift(-1)  # predict next day close
    df = df.dropna().reset_index(drop=True)

    # select numeric columns except date and target
    drop_cols = ['date','target']
    feature_cols = [c for c in df.columns if c not in drop_cols and df[c].dtype in [float, int]]
    print("Using features:", feature_cols)

    X = df[feature_cols].values
    y = df['target'].values

    # scale features (helps some models)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # time-based split
    split = int(len(X_scaled) * 0.8)
    X_train, X_test = X_scaled[:split], X_scaled[split:]
    y_train, y_test = y[:split], y[split:]

    print(f"Train rows: {len(X_train)}, Test rows: {len(X_test)}")

    # Train RandomForest
    model = RandomForestRegressor(n_estimators=300, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    mae = mean_absolute_error(y_test, preds)
    rmse = mean_squared_error(y_test, preds) ** 0.5
    r2 = r2_score(y_test, preds)

    print(f"MAE: {mae:.3f}")
    print(f"RMSE: {rmse:.3f}")
    print(f"R2: {r2:.3f}")

    # Save scaler and model as a dict
    os.makedirs(os.path.join(BASE_DIR), exist_ok=True)
    joblib.dump({'model': model, 'scaler': scaler, 'features': feature_cols}, MODEL_OUT)
    print("Saved model to:", MODEL_OUT)

    # save metrics and importances
    metrics = {'mae': float(mae), 'rmse': float(rmse), 'r2': float(r2)}
    with open(METRICS_OUT, 'w') as f:
        json.dump(metrics, f, indent=2)
    fi = pd.DataFrame({
        'feature': feature_cols,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    fi.to_csv(IMPORTANCE_OUT, index=False)
    print("Saved feature importances to:", IMPORTANCE_OUT)
    print("Saved metrics to:", METRICS_OUT)

if __name__ == "__main__":
    train_and_save()
