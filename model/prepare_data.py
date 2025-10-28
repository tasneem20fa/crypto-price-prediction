# model/prepare_data.py
import os
import pandas as pd
import ta

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
INPUT = os.path.join(BASE_DIR, "..", "data", "BTC-2017min.csv")
OUT_DAILY = os.path.join(BASE_DIR, "..", "data", "BTC_daily_2017.csv")
OUT_FEATURES = os.path.join(BASE_DIR, "..", "data", "features_enhanced.csv")

def prepare():
    print("Loading:", INPUT)
    df = pd.read_csv(INPUT, parse_dates=['date'])
    df = df.sort_values('date').set_index('date')

    # Resample minute -> daily OHLCV
    daily = df.resample('D').agg({
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last',
        'Volume BTC': 'sum',
        'Volume USD': 'sum'
    }).dropna()

    # Basic features
    daily['return'] = daily['close'].pct_change()
    daily['MA7'] = daily['close'].rolling(7).mean()
    daily['MA30'] = daily['close'].rolling(30).mean()
    daily['MA50'] = daily['close'].rolling(50).mean()
    daily['MA100'] = daily['close'].rolling(100).mean()
    daily['MA200'] = daily['close'].rolling(200).mean()
    daily['volatility7'] = daily['close'].rolling(7).std()
    daily['volatility30'] = daily['close'].rolling(30).std()
    daily['lag1_close'] = daily['close'].shift(1)
    daily['day_of_week'] = daily.index.dayofweek

    # Use `ta` library for RSI, MACD, ATR
    # ta requires a Series with names 'high','low','close','volume'
    # RSI
    try:
        daily['rsi14'] = ta.momentum.RSIIndicator(daily['close'], window=14).rsi()
        macd = ta.trend.MACD(daily['close'])
        daily['macd'] = macd.macd()
        daily['macd_signal'] = macd.macd_signal()
        daily['macd_diff'] = macd.macd_diff()
        daily['atr14'] = ta.volatility.AverageTrueRange(
            daily['high'], daily['low'], daily['close'], window=14).average_true_range()
    except Exception as e:
        print("ta indicators failed:", e)
        # keep going — indicators optional

    # Momentum / percent changes
    daily['roc5'] = daily['close'].pct_change(5)
    daily['roc10'] = daily['close'].pct_change(10)

    # Drop rows with NaNs (from rolling windows)
    daily = daily.dropna().reset_index()

    # Choose features for modeling
    feature_columns = [
        'date','open','high','low','close','Volume BTC',
        'MA7','MA30','MA50','MA100','MA200',
        'volatility7','volatility30',
        'lag1_close','day_of_week',
        'rsi14','macd','macd_signal','macd_diff','atr14',
        'roc5','roc10'
    ]
    # Keep only columns that exist (ta might not have produced some)
    feature_columns = [c for c in feature_columns if c in daily.columns]

    features = daily[feature_columns].copy()
    features.to_csv(OUT_FEATURES, index=False)
    daily.to_csv(OUT_DAILY, index=False)

    print("Saved:", OUT_DAILY)
    print("Saved:", OUT_FEATURES)
    print("Rows:", len(features))
    print("Columns:", features.columns.tolist())

if __name__ == "__main__":
    prepare()
