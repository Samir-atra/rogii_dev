"""Unit tests for the improved Residual Conv1D + SE model architecture.

Validates model instantiation, layer output shapes, BatchNormalization presence,
forward pass tensor shapes, loss calculation, and JAX float64 denormalization.
"""

import os
import unittest

os.environ["KERAS_BACKEND"] = "jax"
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

import jax
import jax.numpy as jnp
import keras
from keras import Model, layers, regularizers
import numpy as np

jax.config.update("jax_enable_x64", True)

FEATURE_COLS = ["MD", "X", "Y", "Z", "GR", "TVT_input"]
INPUT_DIM = len(FEATURE_COLS)


def se_block(x, ratio=8):
    """Applies Squeeze-and-Excitation channel attention.

    Args:
        x: Input tensor of shape (batch, steps, channels).
        ratio: Reduction ratio for the SE bottleneck.

    Returns:
        Tensor with channel-wise recalibration applied.
    """
    channels = x.shape[-1]
    se = layers.GlobalAveragePooling1D()(x)
    se = layers.Dense(max(channels // ratio, 4), activation="relu")(se)
    se = layers.Dense(channels, activation="sigmoid")(se)
    se = layers.Reshape((1, channels))(se)
    return layers.Multiply()([x, se])


def residual_conv1d_block(x, filters, kernel_size, dropout_rate=0.2):
    """Builds a residual Conv1D block with BatchNorm and SE attention.

    Args:
        x: Input tensor.
        filters: Number of output filters.
        kernel_size: Convolution kernel size.
        dropout_rate: Dropout rate after the block.

    Returns:
        Output tensor of the residual block.
    """
    shortcut = x
    y = layers.Conv1D(
        filters,
        kernel_size,
        padding="causal",
        kernel_initializer="he_normal",
        kernel_regularizer=regularizers.l2(1e-4),
    )(x)
    y = layers.BatchNormalization()(y)
    y = layers.LeakyReLU(alpha=0.1)(y)
    y = layers.Conv1D(
        filters,
        kernel_size,
        padding="causal",
        kernel_initializer="he_normal",
        kernel_regularizer=regularizers.l2(1e-4),
    )(y)
    y = layers.BatchNormalization()(y)
    y = se_block(y)

    if shortcut.shape[-1] != filters:
        shortcut = layers.Conv1D(
            filters,
            1,
            padding="same",
            kernel_initializer="he_normal",
            kernel_regularizer=regularizers.l2(1e-4),
        )(shortcut)

    y = layers.Add()([shortcut, y])
    y = layers.LeakyReLU(alpha=0.1)(y)
    y = layers.Dropout(dropout_rate)(y)
    return y


def build_improved_conv1d_model(input_dim: int = INPUT_DIM) -> Model:
    """Builds and compiles the improved Residual Conv1D + SE model.

    Args:
        input_dim: Number of feature dimensions per sequence step.

    Returns:
        Compiled Keras Model.
    """
    inputs = layers.Input(shape=(1, input_dim))

    x = residual_conv1d_block(inputs, filters=64, kernel_size=5, dropout_rate=0.2)
    x = residual_conv1d_block(x, filters=128, kernel_size=5, dropout_rate=0.2)
    x = residual_conv1d_block(x, filters=128, kernel_size=7, dropout_rate=0.2)

    x = layers.Flatten()(x)
    x = layers.Dense(
        128,
        kernel_initializer="he_normal",
        kernel_regularizer=regularizers.l2(1e-4),
    )(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU(alpha=0.1)(x)
    x = layers.Dropout(0.2)(x)
    x = layers.Dense(
        64,
        kernel_initializer="he_normal",
        kernel_regularizer=regularizers.l2(1e-4),
    )(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU(alpha=0.1)(x)
    x = layers.Dropout(0.1)(x)
    outputs = layers.Dense(1, activation="linear")(x)

    model = Model(inputs=inputs, outputs=outputs)

    optimizer = keras.optimizers.AdamW(
        learning_rate=5e-4,
        weight_decay=1e-4,
        global_clipnorm=1.0,
    )
    model.compile(
        optimizer=optimizer,
        loss="mse",
        metrics=[keras.metrics.RootMeanSquaredError(name="rmse")],
    )
    return model


def denormalize_predictions(
    y_norm: np.ndarray,
    target_mean: float,
    target_std: float,
) -> jnp.ndarray:
    """Denormalizes predictions using JAX float64 on CPU.

    Args:
        y_norm: Normalized prediction array.
        target_mean: Training target mean used for scaling.
        target_std: Training target standard deviation used for scaling.

    Returns:
        Denormalized predictions as a JAX array.
    """
    cpu_dev = jax.devices("cpu")[0]
    y_norm_jax = jax.device_put(y_norm.astype(np.float64), cpu_dev)
    mean_jax = jax.device_put(target_mean, cpu_dev)
    std_jax = jax.device_put(target_std, cpu_dev)
    return (y_norm_jax * std_jax) + mean_jax


class TestImprovedConv1DModel(unittest.TestCase):
    """Test suite for improved Conv1D model architecture."""

    def setUp(self):
        """Builds a fresh model instance for each test."""
        self.model = build_improved_conv1d_model()

    def test_model_instantiation(self):
        """Tests that the model builds without errors."""
        self.assertIsInstance(self.model, Model)
        self.assertEqual(self.model.input_shape, (None, 1, INPUT_DIM))
        self.assertEqual(self.model.output_shape, (None, 1))

    def test_batch_normalization_layers_present(self):
        """Tests that BatchNormalization layers exist in the model."""
        bn_layers = [
            layer for layer in self.model.layers if isinstance(layer, layers.BatchNormalization)
        ]
        self.assertGreaterEqual(len(bn_layers), 5)
        for bn in bn_layers:
            self.assertTrue(bn.trainable)
            self.assertEqual(bn.axis, -1)

    def test_leaky_relu_and_se_components(self):
        """Tests presence of LeakyReLU, Add, and Multiply (SE) layers."""
        layer_types = {type(layer).__name__ for layer in self.model.layers}
        self.assertIn("LeakyReLU", layer_types)
        self.assertIn("Add", layer_types)
        self.assertIn("Multiply", layer_types)
        self.assertIn("GlobalAveragePooling1D", layer_types)

    def test_forward_pass_output_shape(self):
        """Tests forward pass produces correct output tensor shape."""
        batch_size = 8
        dummy_input = np.random.randn(batch_size, 1, INPUT_DIM).astype(np.float32)
        output = self.model(dummy_input, training=False)
        self.assertEqual(tuple(output.shape), (batch_size, 1))

    def test_predict_output_shape(self):
        """Tests model.predict output shape on a batch."""
        dummy_input = np.zeros((16, 1, INPUT_DIM), dtype=np.float32)
        preds = self.model.predict(dummy_input, verbose=0)
        self.assertEqual(preds.shape, (16, 1))

    def test_loss_calculation(self):
        """Tests that compiled loss can be computed on synthetic data."""
        x = np.random.randn(32, 1, INPUT_DIM).astype(np.float32)
        y = np.random.randn(32).astype(np.float32)
        metrics = self.model.evaluate(x, y, verbose=0)
        if isinstance(metrics, list):
            loss = metrics[0]
        else:
            loss = metrics
        self.assertIsInstance(loss, float)
        self.assertFalse(np.isnan(loss))
        self.assertGreater(loss, 0.0)

    def test_training_step_updates_weights(self):
        """Tests that one training step runs and updates trainable weights."""
        x = np.random.randn(16, 1, INPUT_DIM).astype(np.float32)
        y = np.random.randn(16).astype(np.float32)
        weights_before = self.model.get_weights()
        self.model.fit(x, y, epochs=1, batch_size=8, verbose=0)
        weights_after = self.model.get_weights()
        changed = any(
            not np.allclose(wb, wa)
            for wb, wa in zip(weights_before, weights_after)
            if wb.size > 0
        )
        self.assertTrue(changed)

    def test_jax_float64_denormalization(self):
        """Tests JAX float64 denormalization round-trip precision."""
        target_mean = -2100.0
        target_std = 45.5
        y_raw_true = np.array([-2145.5, -2100.0, -2054.5], dtype=np.float64)
        y_norm = (y_raw_true - target_mean) / target_std

        y_denorm = denormalize_predictions(y_norm, target_mean, target_std)
        err = jnp.abs(y_denorm - y_raw_true)
        self.assertLess(float(jnp.max(err)), 1e-12)

    def test_optimizer_configuration(self):
        """Tests AdamW optimizer hyperparameters."""
        optimizer = self.model.optimizer
        self.assertIsInstance(optimizer, keras.optimizers.AdamW)
        lr = float(keras.ops.convert_to_numpy(optimizer.learning_rate))
        self.assertAlmostEqual(lr, 5e-4, places=6)


if __name__ == "__main__":
    unittest.main()
