"""Deep Echo State Network (Deep ESN) implementation using JAX.

This module provides the DeepESN class and state generation utilities for sequence
and wellbore log modeling, utilizing multi-layer reservoir architectures.

All linear algebra, matrix operations, spectral radius scaling, and state updates
are implemented using JAX for high precision and performance.
"""

from typing import Dict, List, Optional, Tuple
import jax
import jax.numpy as jnp
from tqdm.auto import trange
import numpy as np


class DeepESN:
    """Multi-layer Deep Echo State Network with JAX backend.

    Attributes:
        input_dim: Dimensionality of the input feature vectors.
        num_layers: Number of stacked reservoir layers.
        reservoir_size: Number of units (neurons) in each reservoir layer.
        spectral_radius: Target spectral radius for reservoir weight matrices.
        leak_rate: Leaking rate for state updates (between 0.0 and 1.0).
        input_scaling: Scaling factor applied to initial input weight matrices.
        sparsity: Fraction of zeros in reservoir connectivity matrices.
        ridge_alpha: L2 regularization strength for the linear readout solver.
    """

    def __init__(
        self,
        input_dim: int,
        num_layers: int = 3,
        reservoir_size: int = 100,
        spectral_radius: float = 0.95,
        leak_rate: float = 0.3,
        input_scaling: float = 0.5,
        sparsity: float = 0.1,
        ridge_alpha: float = 1e-3,
        seed: int = 42,
    ) -> None:
        """Initializes the DeepESN reservoir parameters and generates fixed weights.

        Args:
            input_dim: Number of input features per timestep.
            num_layers: Number of stacked reservoir layers.
            reservoir_size: Number of units in each reservoir layer.
            spectral_radius: Maximum absolute eigenvalue for reservoir weight matrices.
            leak_rate: Leaking rate for reservoir state updates.
            input_scaling: Scaling factor for input weight matrices.
            sparsity: Sparsity density ratio of reservoir weights.
            ridge_alpha: Ridge regression L2 penalty constant.
            seed: Random seed for weight initialization.
        """
        self.input_dim = input_dim
        self.num_layers = num_layers
        self.reservoir_size = reservoir_size
        self.spectral_radius = spectral_radius
        self.leak_rate = leak_rate
        self.input_scaling = input_scaling
        self.sparsity = sparsity
        self.ridge_alpha = ridge_alpha
        self.seed = seed

        self.w_in: List[jnp.ndarray] = []
        self.w_res: List[jnp.ndarray] = []
        self.w_out: Optional[jnp.ndarray] = None

        self._initialize_reservoirs()

    def _initialize_reservoirs(self) -> None:
        """Initializes fixed random input and reservoir weight matrices using JAX.

        Scales the reservoir matrices to guarantee the specified spectral radius.
        """
        key = jax.random.PRNGKey(self.seed)

        print(f"Initializing {self.num_layers} reservoir layers...")
        for layer_idx in range(self.num_layers):
            key, key_in, key_res, key_mask = jax.random.split(key, 4)

            # Input dimension for layer 0 is input_dim; for layer > 0 it is reservoir_size
            in_dim = self.input_dim if layer_idx == 0 else self.reservoir_size

            # Draw uniform random input weights in [-1, 1] scaled by input_scaling
            w_in_raw = jax.random.uniform(
                key_in, shape=(self.reservoir_size, in_dim), minval=-1.0, maxval=1.0
            )
            w_in_layer = w_in_raw * self.input_scaling
            self.w_in.append(w_in_layer)

            # Draw uniform random reservoir weights with sparsity mask
            w_res_raw = jax.random.uniform(
                key_res,
                shape=(self.reservoir_size, self.reservoir_size),
                minval=-1.0,
                maxval=1.0,
            )
            mask = (
                jax.random.uniform(
                    key_mask, shape=(self.reservoir_size, self.reservoir_size)
                )
                < self.sparsity
            )
            w_res_sparse = jnp.where(mask, w_res_raw, 0.0)

            # Calculate spectral radius (max absolute eigenvalue) using numpy/jax eigh
            # Note: convert to numpy for eigen computation on host CPU if needed, then back to JAX
            eigs = np.linalg.eigvals(np.array(w_res_sparse))
            max_eig = np.max(np.abs(eigs))
            if max_eig > 0:
                w_res_scaled = w_res_sparse * (self.spectral_radius / max_eig)
            else:
                w_res_scaled = w_res_sparse

            self.w_res.append(jnp.array(w_res_scaled))
        print("Reservoir initialization complete.")

    def compute_reservoir_states(
        self,
        X: jnp.ndarray,
        initial_states: Optional[List[jnp.ndarray]] = None,
    ) -> jnp.ndarray:
        """Computes multi-layer reservoir state representations using jax.lax.scan.

        Args:
            X: Input matrix of shape (num_samples, input_dim).
            initial_states: Optional list of initial state vectors for each reservoir layer.

        Returns:
            State matrix S of shape (num_samples, input_dim + num_layers * reservoir_size).
        """
        states_matrix, _ = self._compute_reservoir_states_with_final(
            X, initial_states=initial_states
        )
        return states_matrix

    def _compute_reservoir_states_with_final(
        self,
        X: jnp.ndarray,
        initial_states: Optional[List[jnp.ndarray]] = None,
    ) -> Tuple[jnp.ndarray, List[jnp.ndarray]]:
        """Computes reservoir states and returns final states for sequence continuity.

        Args:
            X: Input matrix of shape (num_samples, input_dim).
            initial_states: Optional list of initial state vectors for each layer.

        Returns:
            Tuple of (states_matrix, final_layer_states).
        """
        if initial_states is None:
            init_states = tuple(
                jnp.zeros((self.reservoir_size,), dtype=X.dtype)
                for _ in range(self.num_layers)
            )
        else:
            init_states = tuple(initial_states)

        # Cast weights to match input dtype to keep jax.lax.scan carry dtypes consistent
        dtype = X.dtype
        w_in_tuple = tuple(w.astype(dtype) for w in self.w_in)
        w_res_tuple = tuple(w.astype(dtype) for w in self.w_res)
        leak_rate = jnp.array(self.leak_rate, dtype=dtype)
        num_layers = self.num_layers

        def step_fn(states, x_t):
            new_states = []
            current_in = x_t
            for l in range(num_layers):
                linear_comb = jnp.dot(w_in_tuple[l], current_in) + jnp.dot(
                    w_res_tuple[l], states[l]
                )
                h_tilde = jnp.tanh(linear_comb)
                h_l = (jnp.array(1.0, dtype=dtype) - leak_rate) * states[l] + leak_rate * h_tilde
                new_states.append(h_l)
                current_in = h_l
            combined_t = jnp.concatenate([x_t] + new_states, axis=0)
            return tuple(new_states), combined_t

        final_states, states_matrix = jax.lax.scan(step_fn, init_states, X)
        return states_matrix, list(final_states)

    def _evaluate_predictions(
        self,
        X: jnp.ndarray,
        chunk_size: int = 100000,
    ) -> jnp.ndarray:
        """Computes linear readout predictions in memory-efficient chunks.

        Args:
            X: Input feature matrix of shape (num_samples, input_dim).
            chunk_size: Number of samples per evaluation chunk.

        Returns:
            Predicted output target vector of shape (num_samples,).
        """
        num_samples = X.shape[0]
        if num_samples <= chunk_size:
            states = self.compute_reservoir_states(X)
            return jnp.dot(states, self.w_out)

        preds_list = []
        init_states = None

        num_chunks = (num_samples + chunk_size - 1) // chunk_size
        for c in range(num_chunks):
            start_i = c * chunk_size
            end_i = min(start_i + chunk_size, num_samples)
            X_chunk = X[start_i:end_i]

            chunk_states, last_states = self._compute_reservoir_states_with_final(
                X_chunk, initial_states=init_states
            )
            pred_chunk = jnp.dot(chunk_states, self.w_out)
            preds_list.append(pred_chunk)
            init_states = last_states

        return jnp.concatenate(preds_list, axis=0)

    def fit(
        self,
        X: jnp.ndarray,
        y: jnp.ndarray,
    ) -> Dict[str, List[float]]:
        """Fits the linear readout weights using a memory-efficient Ridge Regression solver.

        This method avoids materializing the full state matrix `S` by computing the
        components for the normal equation (`S.T @ S` and `S.T @ y`) in chunks.
        This is crucial for large datasets that do not fit in memory.

        The analytical solution to Ridge Regression is:

        w_out = (S.T @ S + alpha * I)^-1 @ S.T @ y

        Args:
            X: Standardized input feature matrix of shape (num_samples, input_dim).
            y: Standardized target vector of shape (num_samples,).

        Returns:
            A dictionary containing the final training loss, mimicking a Keras history object.
        
        Raises:
            ValueError: If the model has already been fitted.
        """
        if self.w_out is not None:
            raise ValueError("Model has already been fitted. Re-initialize to train again.")

        print("\nStarting DeepESN training with memory-efficient Ridge solver...")

        num_samples = X.shape[0]
        chunk_size = 100000  # Use the same chunk size as prediction
        num_features = self.input_dim + self.num_layers * self.reservoir_size

        # Initialize accumulators for S.T @ S and S.T @ y
        StS = jnp.zeros((num_features, num_features), dtype=X.dtype)
        Sty = jnp.zeros((num_features, 1), dtype=X.dtype)

        if y.ndim == 1:
            y = y.reshape(-1, 1)

        init_states = None
        num_chunks = (num_samples + chunk_size - 1) // chunk_size

        print(f"Processing {num_samples} samples in {num_chunks} chunks...")
        for c in trange(num_chunks, desc="Training Chunks"):
            start_i = c * chunk_size
            end_i = min(start_i + chunk_size, num_samples)
            X_chunk = X[start_i:end_i]
            y_chunk = y[start_i:end_i]

            # Compute states for the current chunk
            S_chunk, last_states = self._compute_reservoir_states_with_final(
                X_chunk, initial_states=init_states
            )

            # Accumulate S.T @ S and S.T @ y
            StS += S_chunk.T @ S_chunk
            Sty += S_chunk.T @ y_chunk

            # Pass final states of this chunk as initial states for the next
            init_states = last_states

        # Add the regularization term
        # The regularization term should be scaled by the number of samples
        # to be consistent with many ML frameworks.
        I = jnp.identity(num_features, dtype=X.dtype)
        A = StS + self.ridge_alpha * num_samples * I

        # Solve the linear system A * w_out = Sty
        print("\nSolving for readout weights via JAX linear algebra solver...")
        w_out_solved = jnp.linalg.solve(A, Sty)

        # Store the flattened weights
        self.w_out = w_out_solved.flatten()

        # Calculate final training loss (RMSE) to return a history-like object
        print("Calculating final training RMSE...")
        y_pred = self._evaluate_predictions(X)
        
        # Ensure y is flat for metric calculation
        y_true = y.flatten()
        
        training_rmse = jnp.sqrt(jnp.mean((y_pred - y_true) ** 2))
        print(f"Training finished. Final Training RMSE: {training_rmse:.4f}")

        return {"loss": [float(training_rmse)], "rmse": [float(training_rmse)]}

    def predict(self, X: jnp.ndarray, chunk_size: int = 100000) -> jnp.ndarray:
        """Predicts output targets using the trained readout weights in memory-efficient chunks.

        Args:
            X: Input feature matrix of shape (num_samples, input_dim).
            chunk_size: Sample chunk size for evaluation.

        Returns:
            Predicted output target vector of shape (num_samples,).

        Raises:
            ValueError: If the model has not been fitted prior to prediction.
        """
        if self.w_out is None:
            raise ValueError("DeepESN model has not been fitted yet. Call fit() first.")

        return self._evaluate_predictions(X, chunk_size=chunk_size)
