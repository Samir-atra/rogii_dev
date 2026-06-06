import numpy as np
import jax.numpy as jnp
import pytest

def test_precision_conversion():
    # Simulate high precision data
    high_prec = np.array([1.123456789, 2.987654321], dtype=np.float64)
    # Convert to float16
    low_prec = high_prec.astype(np.float16)
    
    assert low_prec.dtype == np.float16
    # Precision loss is expected
    assert not np.array_equal(high_prec, low_prec.astype(np.float64))
    
    # Check JAX conversion
    jax_low = jnp.array(high_prec, dtype=jnp.float16)
    assert jax_low.dtype == jnp.float16

if __name__ == "__main__":
    pytest.main([__file__])
