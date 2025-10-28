# model/train_and_compare.py
import os, sys, joblib, json
import pandas as pd, numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb
import lightgbm as lgb

BASE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(BASE, "..", "data", "features_enhanced.csv")
OUT_MODEL = os.path.join(BASE, "best_model.pkl")
OUT_METRICS = os.path.join(BASE, "compare_metrics.json")

if not os.path.exists(DATA):
    print("Run prepare_data.py first")
    sys.exit(1)

df = pd.read_csv(DATA, parse_dates=['date']).sort_values('date').reset_index(drop=True)
df['target'] = df['close'].shift(-1)
df = df.dropna().reset_index(drop=True)

drop = ['date','target','close']  # keep close if you want ratios only
feature_cols = [c for c in df.columns if c not in drop and np.issubdtype(df[c].dtype, np.number)]
X = df[feature_cols].values
y = df['target'].values

scaler = StandardScaler()
Xs = scaler.fit_transform(X)

tss = TimeSeriesSplit(n_splits=5)
models = {
    'RandomForest': RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1),
    'XGBoost': xgb.XGBRegressor(n_estimators=200, learning_rate=0.05, random_state=42, verbosity=0),
    'LightGBM': lgb.LGBMRegressor(n_estimators=200, learning_rate=0.05, random_state=42)
}

results = {}
for name, model in models.items():
    maes, rmses, r2s = [], [], []
    for train_idx, test_idx in tss.split(Xs):
        Xtr, Xte = Xs[train_idx], Xs[test_idx]
        ytr, yte = y[train_idx], y[test_idx]
        model.fit(Xtr, ytr)
        preds = model.predict(Xte)
        maes.append(mean_absolute_error(yte, preds))
        # compute RMSE manually (compatible with all sklearn versions)
        mse = mean_squared_error(yte, preds)
        rmses.append(mse ** 0.5)
        r2s.append(r2_score(yte, preds))
    results[name] = {
        'mae_mean': float(np.mean(maes)),
        'rmse_mean': float(np.mean(rmses)),
        'r2_mean': float(np.mean(r2s))
    }
    print(f"{name}: MAE={np.mean(maes):.2f}, RMSE={np.mean(rmses):.2f}, R2={np.mean(r2s):.3f}")

# Pick best by RMSE (or MAE)
best_name = min(results.keys(), key=lambda k: results[k]['rmse_mean'])
best_model = models[best_name]
# Retrain best model on full set
best_model.fit(Xs, y)

joblib.dump({'model': best_model, 'scaler': scaler, 'features': feature_cols}, OUT_MODEL)
with open(OUT_METRICS, 'w') as f:
    json.dump({'results': results, 'best': best_name}, f, indent=2)
print("Best:", best_name, "saved to", OUT_MODEL)
