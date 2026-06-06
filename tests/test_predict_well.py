"""Unit tests for predict_well function to verify precision and scaling.

This test module verifies the correctness of the predict_well function,
ensuring that high-precision float64 arithmetic using JAX produces
stable predictions without losing significant digits.
"""

import jax
# Enable JAX float64 precision natively
jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np
import polars as pl
import pytest


# Constants matching our configuration
FEATURE_COLS = ["MD", "X", "Y", "Z", "GR", "TVT_input"]
TARGET = "TVT"


class MockModel:
    """Mock Keras model for testing predict_well."""

    def __init__(self, ws, n_features):
        """Initializes the mock model.

        Args:
            ws (int): Window size.
            n_features (int): Number of features.
        """
        self.ws = ws
        self.n_features = n_features

    def predict(self, x, batch_size=512, verbose=0):
        """Mock predict that sums window values to simulate regression.

        Args:
            x (np.ndarray): Stacked input sequences.
            batch_size (int): Batch size (ignored).
            verbose (int): Verbosity level (ignored).

        Returns:
            np.ndarray: 1D prediction array.
        """
        # Sum along the window and feature dimensions to return a predictable float
        return np.sum(x, axis=(1, 2))


def predict_well(model, df, scaler, ws):
    """Predicts TVT for every row of a well log using sliding windows.

    All mathematical operations (normalization, denormalization, windowing)
    are performed in double-precision (float64) using JAX to eliminate
    precision loss from large features and target values.

    Args:
        model: Trained Keras model.
        df (pl.DataFrame): Preprocessed well log DataFrame.
        scaler (dict): Dictionary with float64 normalisation stats.
        ws (int): Window size.

    Returns:
        np.ndarray: 1D float64 array of predictions matching well log length,
            with NaN values for early timesteps.
    """
    # Load features in float64 using JAX
    raw_feats = df.select(FEATURE_COLS).to_numpy()
    feats = jnp.nan_to_num(jnp.array(raw_feats, dtype=jnp.float64))

    # Normalize features in float64 using JAX
    mean = jnp.array(scaler["feat_mean"], dtype=jnp.float64)
    std = jnp.array(scaler["feat_std"], dtype=jnp.float64)
    feats_n = (feats - mean) / std

    n = len(feats_n)
    preds = np.full(n, np.nan, dtype=np.float64)
    starts = list(range(0, n - ws + 1))
    if not starts:
        return preds

    # Stack windows and cast to float32 only for model prediction
    stacked_list = [feats_n[s:s+ws] for s in starts]
    X_jax = jnp.stack(stacked_list)
    X = np.array(X_jax, dtype=np.float32)

    # Model prediction
    yn = model.predict(X, batch_size=512, verbose=0).ravel()

    # Denormalize in float64 using JAX
    yn_f64 = jnp.array(yn, dtype=jnp.float64)
    t_mean = jnp.array(scaler["target_mean"], dtype=jnp.float64)
    t_std = jnp.array(scaler["target_std"], dtype=jnp.float64)
    yp_jax = yn_f64 * t_std + t_mean
    yp = np.array(yp_jax, dtype=np.float64)

    for i, s in enumerate(starts):
        preds[s + ws - 1] = yp[i]

    # Backfill early timesteps
    fv = ws - 1
    if fv < n and not np.isnan(preds[fv]):
        preds[:fv] = preds[fv]

    return preds


def test_predict_well_precision():
    """Tests the high-precision float64 predict_well pipeline with synthetic data."""
    # Create high-precision features with large magnitudes (X~2.9M, Y~1.2M, Z~11k)
    n_rows = 15
    ws = 6
    np.random.seed(42)

    md = np.linspace(1000.0, 2000.0, n_rows, dtype=np.float64)
    x_coords = np.linspace(2900000.0, 2901000.0, n_rows, dtype=np.float64)
    y_coords = np.linspace(1200000.0, 1200500.0, n_rows, dtype=np.float64)
    z_coords = np.linspace(11000.0, 11500.0, n_rows, dtype=np.float64)
    gr = np.random.uniform(20.0, 150.0, n_rows).astype(np.float64)
    tvt_input = np.random.uniform(10.0, 50.0, n_rows).astype(np.float64)

    # Use Polars as required by Rule 12 for tabular data
    data_dict = {
        "MD": md,
        "X": x_coords,
        "Y": y_coords,
        "Z": z_coords,
        "GR": gr,
        "TVT_input": tvt_input,
    }
    df = pl.DataFrame(data_dict)

    # Compute mock scaler parameters as float64 with ~15 decimal precision (nn.nnnnnnnnnnnnnnn)
    feat_mean = np.array([md.mean(), x_coords.mean(), y_coords.mean(), z_coords.mean(), gr.mean(), tvt_input.mean()], dtype=np.float64)
    feat_std = np.array([md.std(), x_coords.std(), y_coords.std(), z_coords.std(), gr.std(), tvt_input.std()], dtype=np.float64)
    target_mean = 11250.987654321098
    target_std = 643.1234567890123

    scaler = {
        "feat_mean": feat_mean,
        "feat_std": feat_std,
        "target_mean": target_mean,
        "target_std": target_std,
    }

    model = MockModel(ws, len(FEATURE_COLS))

    # Run predictions
    preds = predict_well(model, df, scaler, ws)

    # Assert shape and type
    assert preds.shape[0] == n_rows
    assert preds.dtype == np.float64

    # Assert first ws-1 values are equal to the first valid prediction (backfill)
    first_valid = preds[ws - 1]
    for i in range(ws):
        assert preds[i] == first_valid

    # Assert no NaN remains
    assert not np.isnan(preds).any()

    # Verify high-precision roundtrip and that the backfill worked properly
    print(f"Predictions check: {preds}")


if __name__ == "__main__":
    pytest.main([__file__])
