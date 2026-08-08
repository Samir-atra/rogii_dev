"""Unit test for robust log-normalization and inversion in ROGII well dataset.

Tests column-wise shift + log1p + standardization and target inversion using JAX.
"""

import glob
import os
import unittest
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np
import polars as pl

FEATURE_COLS = ["MD", "X", "Y", "Z", "GR", "TVT_input"]
TARGET = "TVT"
DATA_DIR = "/home/samer/Documents/competitions/ROGII/dataset"


def log_transform_features(
    X_jax: jnp.ndarray,
    shifts: jnp.ndarray = None,
    means: jnp.ndarray = None,
    stds: jnp.ndarray = None,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Applies shifted log1p transform followed by Z-score standardization.

    Args:
        X_jax (jnp.ndarray): 2D array of features [samples, num_features].
        shifts (jnp.ndarray, optional): Precomputed feature shift values.
        means (jnp.ndarray, optional): Precomputed log-feature means.
        stds (jnp.ndarray, optional): Precomputed log-feature stds.

    Returns:
        tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
            Normalized features, shifts, means, stds.
    """
    if shifts is None:
        # Shift each feature column so min value is non-negative (min >= 0)
        min_vals = jnp.min(X_jax, axis=0)
        shifts = jnp.where(min_vals < 0, -min_vals, 0.0)

    X_shifted = X_jax + shifts
    X_log = jnp.log1p(X_shifted)

    if means is None:
        means = jnp.mean(X_log, axis=0)
    if stds is None:
        stds = jnp.std(X_log, axis=0)
        stds = jnp.where(stds == 0, 1.0, stds)

    X_norm = (X_log - means) / stds
    return X_norm, shifts, means, stds


def log_transform_target(
    y_jax: jnp.ndarray, mean: float = None, std: float = None
) -> tuple[jnp.ndarray, float, float]:
    """Applies log1p transform and optional Z-score standardization to target.

    Args:
        y_jax (jnp.ndarray): 1D array of target values.
        mean (float, optional): Precomputed log-target mean.
        std (float, optional): Precomputed log-target std.

    Returns:
        tuple[jnp.ndarray, float, float]: Normalized target, mean, std.
    """
    y_log = jnp.log1p(y_jax)
    if mean is None:
        mean = float(jnp.mean(y_log))
    if std is None:
        std = float(jnp.std(y_log))
        if std == 0:
            std = 1.0

    y_norm = (y_log - mean) / std
    return y_norm, mean, std


def inverse_log_target(y_norm_jax: jnp.ndarray, mean: float, std: float) -> jnp.ndarray:
    """Inverts normalized log1p target back to raw scale.

    Args:
        y_norm_jax (jnp.ndarray): Normalized target predictions.
        mean (float): Log-target mean.
        std (float): Log-target std.

    Returns:
        jnp.ndarray: Raw scale predictions.
    """
    y_log = (y_norm_jax * std) + mean
    return jnp.expm1(y_log)


class TestLogNormalizationFix(unittest.TestCase):
    """Test suite for validating log-normalization fix."""

    def test_log_normalization_and_inversion(self) -> None:
        """Tests that shifted log-normalization produces no NaNs and reconstructs y_raw."""
        files = sorted(glob.glob(os.path.join(DATA_DIR, "train", "*__horizontal_well.csv")))
        self.assertGreater(len(files), 0, "No training files found.")

        df_list = []
        for f in files[:20]:
            df = pl.read_csv(f, infer_schema_length=10000, ignore_errors=True)
            cols = [c for c in FEATURE_COLS + [TARGET] if c in df.columns]
            exprs = [pl.col(c).cast(pl.Float64, strict=False).alias(c) for c in cols]
            df_cast = df.select(exprs).interpolate().fill_null(strategy="forward").fill_null(strategy="backward").fill_null(0.0)
            df_list.append(df_cast)

        combined_df = pl.concat(df_list, how="diagonal")
        X_raw = combined_df.select(FEATURE_COLS).to_numpy()
        y_raw = combined_df.select(TARGET).to_numpy().ravel()

        X_jax = jnp.array(X_raw, dtype=jnp.float64)
        y_jax = jnp.array(y_raw, dtype=jnp.float64)

        # 1. Transform features
        X_norm, shifts, means, stds = log_transform_features(X_jax)
        self.assertFalse(jnp.isnan(X_norm).any(), "X_norm contains NaNs!")
        self.assertFalse(jnp.isinf(X_norm).any(), "X_norm contains Infs!")

        # 2. Transform target
        y_norm, y_mean, y_std = log_transform_target(y_jax)
        self.assertFalse(jnp.isnan(y_norm).any(), "y_norm contains NaNs!")
        self.assertFalse(jnp.isinf(y_norm).any(), "y_norm contains Infs!")

        # 3. Test exact inverse reconstruction of y_raw
        y_reconstructed = inverse_log_target(y_norm, y_mean, y_std)
        max_diff = float(jnp.max(jnp.abs(y_reconstructed - y_jax)))
        print(f"\n--- Log Normalization Unit Test Success ---")
        print(f"Feature shifts: {shifts}")
        print(f"Max reconstruction error on y_raw: {max_diff:.10e}")
        self.assertLess(max_diff, 1e-6, "Target reconstruction error is too high!")


if __name__ == "__main__":
    unittest.main()
