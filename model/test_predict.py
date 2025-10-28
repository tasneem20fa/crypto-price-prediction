import joblib
import numpy as np
import os

# Load model
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "crypto_model.pkl")
model = joblib.load(MODEL_PATH)

# Create dummy example (open, high, low, Volume BTC, MA7, MA30, lag1_close, day_of_week)
sample = np.array([[27000, 27300, 26800, 1200, 27100, 27200, 26950, 4]])
pred = model.predict(sample)
print("Predicted Close Price:", pred[0])
