"""Unit tests for Deep Echo State Network (Deep ESN) and Z-score normalization.

Verifies reservoir state computation, spectral radius scaling, JAX Ridge Regression
readout solver, and Z-score normalization / denormalization pipeline.
"""

import os
import sys
import pytest
import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO_ROOT, "src"))

from deep_esn import DeepESN


def test_deep_esn_initialization() -> None:
    """Tests DeepESN initialization, weight dimensions, and spectral radius."""
    input_dim = 6
    num_layers = 3
    reservoir_size = 50
    spectral_radius = 0.90

    model = DeepESN(
        input_dim=input_dim,
        num_layers=num_layers,
        reservoir_size=reservoir_size,
        spectral_radius=spectral_radius,
        seed=123,
    )

    assert len(model.w_in) == num_layers
    assert len(model.w_res) == num_layers
    assert model.w_in[0].shape == (reservoir_size, input_dim)
    assert model.w_in[1].shape == (reservoir_size, reservoir_size)

    # Verify spectral radius of layer 0
    w_res_np = np.array(model.w_res[0])
    eigs = np.abs(np.linalg.eigvals(w_res_np))
    assert np.isclose(np.max(eigs), spectral_radius, atol=1e-3)


def test_compute_reservoir_states_shape() -> None:
    """Tests reservoir state generation and output dimensions."""
    num_samples = 100
    input_dim = 6
    num_layers = 4
    reservoir_size = 30

    model = DeepESN(
        input_dim=input_dim,
        num_layers=num_layers,
        reservoir_size=reservoir_size,
    )

    X = jnp.array(np.random.randn(num_samples, input_dim), dtype=jnp.float64)
    states = model.compute_reservoir_states(X)

    expected_dim = input_dim + num_layers * reservoir_size
    assert states.shape == (num_samples, expected_dim)
    assert not jnp.isnan(states).any()


def test_deep_esn_fit_and_predict() -> None:
    """Tests fitting readout weights with JAX Ridge solver and predicting target."""
    num_samples = 200
    input_dim = 4
    num_layers = 2
    reservoir_size = 40

    rng = np.random.default_rng(42)
    X_raw = rng.normal(loc=50.0, scale=15.0, size=(num_samples, input_dim))
    y_raw = X_raw[:, 0] * 2.5 - X_raw[:, 1] * 1.2 + rng.normal(0, 0.1, size=num_samples)

    # Z-Score normalization with JAX
    X_jax = jnp.array(X_raw, dtype=jnp.float64)
    y_jax = jnp.array(y_raw, dtype=jnp.float64)

    feat_mean = jnp.mean(X_jax, axis=0)
    feat_std = jnp.std(X_jax, axis=0)
    target_mean = jnp.mean(y_jax)
    target_std = jnp.std(y_jax)

    X_norm = (X_jax - feat_mean) / feat_std
    y_norm = (y_jax - target_mean) / target_std

    model = DeepESN(
        input_dim=input_dim,
        num_layers=num_layers,
        reservoir_size=reservoir_size,
        ridge_alpha=0.5,
        seed=42,
    )

    model.fit(X_norm, y_norm)
    assert model.w_out is not None

    preds_norm = model.predict(X_norm)
    preds_raw = (preds_norm * target_std) + target_mean

    # Ensure predictions are close to true targets (raw data range ~125, 50 Adam epochs)
    rmse = jnp.sqrt(jnp.mean(jnp.square(preds_raw - y_jax)))
    assert rmse < 1.0
    assert not jnp.isnan(preds_raw).any()


def test_chunked_prediction_equivalence() -> None:
    """Verifies that chunked predictions match full predictions exactly."""
    num_samples = 500
    input_dim = 5
    model = DeepESN(input_dim=input_dim, num_layers=2, reservoir_size=20, seed=42)
    # Use float32 for consistency with the default JAX precision setting
    rng = np.random.default_rng(0)
    X = jnp.array(rng.standard_normal((num_samples, input_dim)), dtype=jnp.float32)
    y = jnp.array(rng.standard_normal(num_samples), dtype=jnp.float32)
    model.fit(X, y)

    pred_full = model.predict(X, chunk_size=5000)
    pred_chunked = model.predict(X, chunk_size=100)

    assert jnp.allclose(pred_full, pred_chunked, atol=1e-5)


if __name__ == "__main__":
    pytest.main([__file__])
