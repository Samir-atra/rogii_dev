
import os
import numpy as np
import polars as pl
import keras
import jax.numpy as jnp
import pickle

# Setup
os.environ["KERAS_BACKEND"] = "jax"
FEATURE_COLS = ["MD", "X", "Y", "Z", "GR", "TVT_input"]
DATA_DIR = "/home/samer/Documents/competitions/ROGII/dataset"
MODEL_PATH = "/home/samer/Documents/competitions/ROGII/outputs/optimized_lstm_model.keras"
SCALER_PATH = "/home/samer/Documents/competitions/ROGII/outputs/opt_scaler_params.pkl"

# Load Model/Scaler
model = keras.saving.load_model(MODEL_PATH)
with open(SCALER_PATH, "rb") as f:
    scaler = pickle.load(f)
MASK_VALUE = scaler.get("mask_value", -10.0)

def predict_well_ws(model, df, scaler, ws):
    raw_feats = df.select(FEATURE_COLS).to_numpy()
    feats = jnp.nan_to_num(jnp.array(raw_feats, dtype=jnp.float64))
    mean = jnp.array(scaler["feat_mean"], dtype=jnp.float64)
    std = jnp.array(scaler["feat_std"], dtype=jnp.float64)
    feats_p = (feats - mean) / std
    
    n = len(feats_p)
    preds = np.full(n, np.nan, dtype=np.float64)
    starts = list(range(0, n - ws + 1))
    
    # Simple check for WS compatibility: The model expects (None, WS, 6)
    # If we pass WS=6 but model was trained on WS=2, it will fail.
    try:
        stacked = jnp.stack([feats_p[s:s+ws] for s in starts])
        X = np.array(stacked, dtype=np.float32)
        yn = model.predict(X, batch_size=512, verbose=0).ravel()
    except Exception as e:
        return f"Error with WS={ws}: {e}"
        
    target_mean = jnp.array(scaler["target_mean"], dtype=jnp.float64)
    target_std = jnp.array(scaler["target_std"], dtype=jnp.float64)
    yp = (jnp.array(yn, dtype=jnp.float64) * target_std) + target_mean
    return np.array(yp).mean()

# Test Well
wid = "000d7d20"
df = pl.read_csv(os.path.join(DATA_DIR, "train", f"{wid}__horizontal_well.csv"))
df = df.with_columns([pl.col(col).interpolate() for col in FEATURE_COLS])

print(f"Testing WS=2: {predict_well_ws(model, df, scaler, 2)}")
print(f"Testing WS=6: {predict_well_ws(model, df, scaler, 6)}")
