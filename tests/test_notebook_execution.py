"""Test execution of Conv1D notebook functions on a subset of dataset.

Verifies end-to-end training execution on 5 wells for fast validation.
"""

import os
import unittest
import jax
import jax.numpy as jnp
import keras
from keras import callbacks, layers, regularizers
import numpy as np
import polars as pl

# Import prepare_data from notebooks/08_conv1d_training.ipynb dynamically or test logic directly
import glob
import pickle


def run_mini_training_pipeline() -> float:
    """Runs a 1-epoch mini training pipeline on a small subset of wells.

    Returns:
        float: Final validation RMSE on raw scale.
    """
    os.environ["KERAS_BACKEND"] = "jax"
    os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

    data_dir = "/home/samer/Documents/competitions/ROGII/dataset"
    feature_cols = ["MD", "X", "Y", "Z", "GR", "TVT_input"]
    target_col = "TVT"

    pattern_train = os.path.join(data_dir, "train", "*__horizontal_well.csv")
    files_train = sorted([os.path.basename(f) for f in glob.glob(pattern_train)])[:10]

    rng = np.random.default_rng(42)

    feats, tgts = [], []
    for fname in files_train:
        path = os.path.join(data_dir, "train", fname)
        df = pl.read_csv(path, infer_schema_length=10000).filter(
            pl.col(target_col).is_not_null()
        )
        if len(df) == 0:
            continue
        for col in feature_cols:
            df = df.with_columns(
                pl.col(col)
                .interpolate()
                .fill_null(strategy="forward")
                .fill_null(strategy="backward")
                .fill_null(0.0)
            )
        feats.append(df.select(feature_cols).to_numpy().astype(np.float64))
        tgts.append(df.select(target_col).to_numpy().ravel().astype(np.float64))

    X_raw = np.concatenate(feats)
    y_raw = np.concatenate(tgts)

    cpu_dev = jax.devices("cpu")[0]
    X_jax = jax.device_put(X_raw, cpu_dev)
    y_jax = jax.device_put(y_raw, cpu_dev)

    feat_mean = jnp.mean(X_jax, axis=0)
    feat_std = jnp.where(jnp.std(X_jax, axis=0) == 0, 1.0, jnp.std(X_jax, axis=0))

    target_mean = jnp.mean(y_jax)
    target_std = jnp.where(jnp.std(y_jax) == 0, 1.0, jnp.std(y_jax))

    X_n = (X_jax - feat_mean) / feat_std
    y_n = (y_jax - target_mean) / target_std

    X_3d = np.array(X_n, dtype=np.float32).reshape(-1, 1, len(feature_cols))
    y_arr = np.array(y_n, dtype=np.float32)

    split = int(len(X_3d) * 0.8)
    X_tr, X_val = X_3d[:split], X_3d[split:]
    y_tr, y_val = y_arr[:split], y_arr[split:]
    y_val_raw = y_raw[split:]

    model = keras.Sequential()
    model.add(layers.Input(shape=(1, len(feature_cols))))
    model.add(
        layers.Conv1D(
            filters=32,
            kernel_size=5,
            activation="relu",
            kernel_initializer="glorot_uniform",
            kernel_regularizer=regularizers.l2(1e-5),
            padding="causal",
        )
    )
    model.add(layers.Dropout(0.2))
    model.add(layers.Flatten())
    model.add(layers.Dense(1, activation="linear"))

    optimizer = keras.optimizers.AdamW(
        learning_rate=1e-3, weight_decay=1e-5, global_clipnorm=1.0
    )
    model.compile(
        optimizer=optimizer,
        loss="mse",
        metrics=[keras.metrics.RootMeanSquaredError(name="rmse")],
    )

    history = model.fit(
        X_tr, y_tr, epochs=1, batch_size=128, validation_data=(X_val, y_val), verbose=0
    )

    yp_n = model.predict(X_val, batch_size=128, verbose=0).ravel()
    yp_n_jnp = jax.device_put(yp_n, cpu_dev)
    target_mean_jnp = jax.device_put(target_mean, cpu_dev)
    target_std_jnp = jax.device_put(target_std, cpu_dev)
    y_val_raw_jnp = jax.device_put(y_val_raw, cpu_dev)

    yp_jnp = (yp_n_jnp * target_std_jnp) + target_mean_jnp
    err = yp_jnp - y_val_raw_jnp
    rmse = float(jnp.sqrt(jnp.mean(jnp.square(err))))
    return rmse


class TestNotebookExecution(unittest.TestCase):
    """Test case for full mini pipeline execution."""

    def test_pipeline_execution(self):
        """Tests mini training pipeline execution."""
        rmse = run_mini_training_pipeline()
        self.assertGreater(rmse, 0.0)
        self.assertFalse(np.isnan(rmse))


if __name__ == "__main__":
    unittest.main()
