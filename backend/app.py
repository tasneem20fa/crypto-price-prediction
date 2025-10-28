from datetime import timedelta
try:
    import ta
except ImportError:
    ta = None

import os
import pandas as pd
import joblib
from flask import Flask, jsonify
from flask_cors import CORS
from sklearn.preprocessing import StandardScaler

app = Flask(__name__)
CORS(app)

# ---------------------------------------------------------
# Paths for model and data
# ---------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "..", "model", "crypto_model_enhanced.pkl")
SCALER_PATH = os.path.join(BASE_DIR, "..", "model", "scaler_enhanced.pkl")
DATA_PATH = os.path.join(BASE_DIR, "..", "data", "features_enhanced.csv")

# ---------------------------------------------------------
# Load model and scaler
# ---------------------------------------------------------
model, scaler, features_list = None, None, []

if os.path.exists(MODEL_PATH):
    saved = joblib.load(MODEL_PATH)
    if isinstance(saved, dict):
        model = saved.get("model", None)
        scaler = saved.get("scaler", None)
        features_list = saved.get("features", [])
    else:
        model = saved
    print("✅ Model loaded from:", MODEL_PATH)
else:
    print("❌ Model file not found at:", MODEL_PATH)

if scaler is None and os.path.exists(SCALER_PATH):
    scaler = joblib.load(SCALER_PATH)
    print("✅ Scaler loaded from:", SCALER_PATH)
elif scaler is None:
    scaler = StandardScaler()
    print("⚠️ Scaler not found; using new StandardScaler()")

# ---------------------------------------------------------
# Load dataset
# ---------------------------------------------------------
if not os.path.exists(DATA_PATH):
    raise FileNotFoundError(f"❌ Data file not found at {DATA_PATH}")

df = pd.read_csv(DATA_PATH)
print(f"✅ Loaded dataset with {len(df)} rows and {df.shape[1]} columns")

# ---------------------------------------------------------
# Define fallback feature list if model didn’t provide one
# ---------------------------------------------------------
if not features_list:
    features_list = [
        "open", "high", "low", "close", "Volume BTC",
        "MA7", "MA30", "MA50", "MA100", "MA200",
        "volatility7", "volatility30", "lag1_close", "day_of_week",
        "rsi14", "macd", "macd_signal", "macd_diff", "atr14", "roc5", "roc10"
    ]

# ---------------------------------------------------------
# Routes
# ---------------------------------------------------------
@app.route("/")
def home():
    return jsonify({"message": "Crypto Price Prediction API is running!"})

# ---------------------------------------------------------
# Predict latest close
# ---------------------------------------------------------
@app.route("/predict_latest", methods=["GET"])
def predict_latest():
    try:
        latest = df.dropna().tail(1)

        if latest.empty:
            return jsonify({"error": "No valid rows found in dataset."}), 400

        missing = [f for f in features_list if f not in latest.columns]
        if missing:
            return jsonify({
                "error": "Model requires additional features not found in data.",
                "missing_features": missing
            }), 400

        X_vals = latest[features_list]
        if len(X_vals) == 0:
            return jsonify({"error": "No valid feature rows for prediction."}), 400

        # Use transform (not fit_transform)
        X_scaled = scaler.transform(X_vals)

        if model is None:
            return jsonify({"error": "Model not loaded properly."}), 500

        y_pred = model.predict(X_scaled)[0]

        return jsonify({
            "predicted_next_close": round(float(y_pred), 2),
            "date_used": latest["date"].values[0] if "date" in latest else None,
            "features_used": features_list
        })

    except Exception as e:
        import traceback
        return jsonify({
            "error": str(e),
            "traceback": traceback.format_exc()
        }), 500
    
    from datetime import timedelta

@app.route("/predict_horizon", methods=["GET"])
def predict_horizon():
    """
    Iteratively predict the next n days (default n=7).
    Query param: ?n=5
    Returns JSON: { "predictions": [ {"date":"YYYY-MM-DD","pred":1234.56}, ... ] }
    """
    try:
        import flask
        # parse n
        n = int(flask.request.args.get("n", 7))
        if n <= 0 or n > 90:
            return jsonify({"error": "n must be between 1 and 90"}), 400

        # load data copy and ensure date parsed
        data = pd.read_csv(DATA_PATH, parse_dates=["date"])
        if data.empty:
            return jsonify({"error": "No history available for forecasting."}), 500

        # make a working copy and ensure sorted by date
        work = data.sort_values("date").copy().reset_index(drop=True)

        # helper to compute features for the whole dataframe (matching prepare_data)
        def compute_rolling_features(df):
            d = df.copy()
            d = d.set_index("date")
            # ensure volume alias exists
            if "Volume BTC" in d.columns:
                d["volume_btc"] = d["Volume BTC"]
            elif "volume_btc" in d.columns:
                d["Volume BTC"] = d["volume_btc"]
            else:
                d["volume_btc"] = 0.0
                d["Volume BTC"] = 0.0

            d["MA7"] = d["close"].rolling(7).mean()
            d["MA30"] = d["close"].rolling(30).mean()
            d["MA50"] = d["close"].rolling(50).mean()
            d["MA100"] = d["close"].rolling(100).mean()
            d["MA200"] = d["close"].rolling(200).mean()
            d["volatility7"] = d["close"].rolling(7).std()
            d["volatility30"] = d["close"].rolling(30).std()
            d["lag1_close"] = d["close"].shift(1)
            d["day_of_week"] = d.index.dayofweek
            d["roc5"] = d["close"].pct_change(5)
            d["roc10"] = d["close"].pct_change(10)
            # optional TA metrics if available
            try:
                if ta is not None:
                    d["rsi14"] = ta.momentum.RSIIndicator(d["close"], window=14).rsi()
                    macd = ta.trend.MACD(d["close"])
                    d["macd"] = macd.macd()
                    d["macd_signal"] = macd.macd_signal()
                    d["macd_diff"] = macd.macd_diff()
                    d["atr14"] = ta.volatility.AverageTrueRange(d["high"], d["low"], d["close"], window=14).average_true_range()
            except Exception:
                pass
            d = d.ffill().bfill().reset_index()
            return d

        # compute initial features
        work = compute_rolling_features(work)

        # ensure features_list is present
        req_feats = features_list.copy()
        missing = [f for f in req_feats if f not in work.columns]
        if missing:
            return jsonify({"error": "Missing features for horizon forecasting", "missing": missing}), 400

        # iterative forecasting
        preds = []
        last_date = pd.to_datetime(work["date"].iloc[-1])
        temp = work.copy()

        for i in range(n):
            # take last row of temp, build feature vector
            X_row = temp[req_feats].tail(1)
            # if empty, abort
            if X_row.empty:
                return jsonify({"error": "Not enough rows to compute features during iteration", "step": i}), 500

            # scale and predict
            X_scaled = scaler.transform(X_row)
            pred = float(model.predict(X_scaled)[0])

            # build next date and append as a new row into temp (so next iteration can compute rolling metrics)
            next_date = last_date + timedelta(days=1)
            # create a new row dict: copy last row and update date and close
            new_row = temp.tail(1).iloc[0].to_dict()
            new_row["date"] = next_date
            new_row["open"] = new_row.get("close", pred)        # approximate open as previous close
            new_row["high"] = max(new_row.get("high", pred), pred)
            new_row["low"] = min(new_row.get("low", pred), pred)
            new_row["close"] = pred
            # ensure Volume BTC stays same or 0
            if "Volume BTC" not in new_row or pd.isna(new_row["Volume BTC"]):
                new_row["Volume BTC"] = new_row.get("volume_btc", 0.0)

            # append to temp DataFrame
            temp = pd.concat([temp, pd.DataFrame([new_row])], ignore_index=True)
            # recompute rolling features on temp (for next iteration)
            temp = compute_rolling_features(temp)

            preds.append({"date": next_date.strftime("%Y-%m-%d"), "predicted_close": round(pred, 2)})

            # update last_date
            last_date = next_date

        return jsonify({"predictions": preds})

    except Exception as e:
        import traceback
        return jsonify({"error": str(e), "traceback": traceback.format_exc()}), 500



# ---------------------------------------------------------
# Run app
# ---------------------------------------------------------
if __name__ == "__main__":
    app.run(debug=True)
