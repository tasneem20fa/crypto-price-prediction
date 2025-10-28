# model/tune_lgbm.py
import os, json, sys
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error
import lightgbm as lgb
import optuna

BASE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(BASE, "..", "data", "features_enhanced.csv")
OUT_JSON = os.path.join(BASE, "lgb_optuna_best.json")

if not os.path.exists(DATA):
    print("ERROR: features_enhanced.csv not found. Run prepare_data.py first.")
    sys.exit(1)

# Load data and prepare target
df = pd.read_csv(DATA, parse_dates=['date']).sort_values('date').reset_index(drop=True)
df['target'] = df['close'].shift(-1)
df = df.dropna().reset_index(drop=True)

# Diagnostic: print shape and dtypes
print("Loaded data shape:", df.shape)
print("Column dtypes:")
print(df.dtypes)

# Choose numeric feature columns (robust)
drop_cols = ['date', 'target', 'close']
feature_cols = []
for c in df.columns:
    if c in drop_cols:
        continue
    # try coerce to numeric if object-like
    if not pd.api.types.is_numeric_dtype(df[c]):
        try:
            df[c] = pd.to_numeric(df[c], errors='coerce')
        except Exception:
            pass
    if pd.api.types.is_numeric_dtype(df[c]):
        feature_cols.append(c)

print("Detected feature columns (count={}):".format(len(feature_cols)))
print(feature_cols)

if len(feature_cols) == 0:
    print("ERROR: No numeric feature columns detected. Check features_enhanced.csv")
    sys.exit(1)

X = df[feature_cols].values
y = df['target'].values

# scale
scaler = StandardScaler()
Xs = scaler.fit_transform(X)

# if dataset is very small, reduce n_splits
n_splits = 5
if len(Xs) < 60:
    n_splits = 3
if len(Xs) < 30:
    n_splits = 2
print("Using TimeSeriesSplit n_splits =", n_splits)
tss = TimeSeriesSplit(n_splits=n_splits)

def objective(trial):
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 50, 500),
        'num_leaves': trial.suggest_int('num_leaves', 8, 128),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2, log=True),
        'min_child_samples': trial.suggest_int('min_child_samples', 5, 200),
        'subsample': trial.suggest_float('subsample', 0.4, 1.0),
        'feature_fraction': trial.suggest_float('feature_fraction', 0.4, 1.0)
    }
    rmses = []
    for train_idx, test_idx in tss.split(Xs):
        Xtr, Xte = Xs[train_idx], Xs[test_idx]
        ytr, yte = y[train_idx], y[test_idx]
        model = lgb.LGBMRegressor(**params, random_state=42, verbose=-1)
        model.fit(Xtr, ytr)
        preds = model.predict(Xte)
        # compute RMSE manually for sklearn compatibility
        mse = mean_squared_error(yte, preds)
        rmse = mse ** 0.5
        rmses.append(rmse)
    return float(np.mean(rmses))

# reduce trials if data small; default 40 -> use 20
n_trials = 20
if len(Xs) < 200:
    n_trials = 20

print(f"Starting Optuna tuning with {n_trials} trials...")
study = optuna.create_study(direction='minimize')
study.optimize(objective, n_trials=n_trials)

best = study.best_params
print("Best params:", best)

# Save best params
with open(OUT_JSON, 'w') as f:
    json.dump(best, f, indent=2)

print("Saved best params to:", OUT_JSON)
