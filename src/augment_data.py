import jax
import jax.numpy as jnp
import polars as pl
import numpy as np
import os
import glob
from pathlib import Path

# Set up paths
INPUT_DIR = Path("dataset/train")
OUTPUT_DIR = Path("dataset_augmented/train")

def sensor_noise_jitter(well_data: pl.DataFrame, key: jax.random.PRNGKey) -> pl.DataFrame:
    """Adds small Gaussian noise to GR, X, Y, Z.
    
    Args:
        well_data: Polars DataFrame of the well.
        key: JAX PRNG key.

    Returns:
        Augmented Polars DataFrame.
    """
    # Convert to JAX for noise
    # We assume 'GR', 'X', 'Y', 'Z' are columns.
    cols = ['GR', 'X', 'Y', 'Z']
    
    # Cast to float, fill nulls
    numeric_data = well_data.select([pl.col(c).cast(pl.Float64).fill_null(0.0) for c in cols])
    data = jnp.array(numeric_data.to_numpy())
    
    noise = jax.random.normal(key, data.shape) * 0.01  # Small magnitude
    augmented_data = data + noise
    
    new_df = well_data.clone()
    new_df = new_df.with_columns([
        pl.Series(col, np.array(augmented_data[:, i])) for i, col in enumerate(cols)
    ])
    return new_df

def gaussian_scaling(well_data: pl.DataFrame, key: jax.random.PRNGKey) -> pl.DataFrame:
    """Scales data along MD by 0.95-1.05.
    
    Args:
        well_data: Polars DataFrame.
        key: JAX PRNG key.
        
    Returns:
        Augmented Polars DataFrame.
    """
    scale = jax.random.uniform(key, (), minval=0.95, maxval=1.05)
    # Scale MD and coordinates? Or just GR? Requirement says "along MD to simulate stratigraphic thickness"
    # Scaling MD changes the depth, scaling the coordinates is also needed to maintain consistency.
    cols = ['MD', 'X', 'Y', 'Z']
    
    # Cast to float, fill nulls
    numeric_data = well_data.select([pl.col(c).cast(pl.Float64).fill_null(0.0) for c in cols])
    data = jnp.array(numeric_data.to_numpy())
    
    # We should only scale MD and X, Y, Z
    augmented_data = data * scale
    
    new_df = well_data.clone()
    new_df = new_df.with_columns([
        pl.Series(col, np.array(augmented_data[:, i])) for i, col in enumerate(cols)
    ])
    return new_df

def magnitude_warping(well_data: pl.DataFrame, key: jax.random.PRNGKey) -> pl.DataFrame:
    """Varies GR intensity using smooth perturbation.
    
    Args:
        well_data: Polars DataFrame.
        key: JAX PRNG key.

    Returns:
        Augmented Polars DataFrame.
    """
    # Cast to float, fill nulls
    gr_data = jnp.array(well_data['GR'].cast(pl.Float64).fill_null(0.0).to_numpy())
    
    # Simple sine wave perturbation for smoothness
    t = jnp.linspace(0, 2 * jnp.pi, len(gr_data))
    perturbation = jax.random.normal(key) * 0.05 * jnp.sin(t)
    
    augmented_gr = gr_data + perturbation
    
    new_df = well_data.clone()
    new_df = new_df.with_columns(pl.Series('GR', np.array(augmented_gr)))
    return new_df

def trajectory_perturbation(well_data: pl.DataFrame, key: jax.random.PRNGKey) -> pl.DataFrame:
    """Applies small constant 3D offset to coordinates.
    
    Args:
        well_data: Polars DataFrame.
        key: JAX PRNG key.

    Returns:
        Augmented Polars DataFrame.
    """
    offset = jax.random.uniform(key, (3,), minval=-0.5, maxval=0.5)
    
    new_df = well_data.clone()
    new_df = new_df.with_columns([
        (pl.col('X').cast(pl.Float64).fill_null(0.0) + offset[0]).alias('X'),
        (pl.col('Y').cast(pl.Float64).fill_null(0.0) + offset[1]).alias('Y'),
        (pl.col('Z').cast(pl.Float64).fill_null(0.0) + offset[2]).alias('Z'),
    ])
    return new_df

def window_slicing(well_data: pl.DataFrame, key: jax.random.PRNGKey) -> pl.DataFrame:
    """Uses random 80% window slice as a new well sample.
    
    Args:
        well_data: Polars DataFrame.
        key: JAX PRNG key.

    Returns:
        Augmented Polars DataFrame.
    """
    n = len(well_data)
    window_size = int(0.8 * n)
    start = jax.random.randint(key, (), 0, n - window_size)
    
    return well_data.slice(int(start), window_size)

def mixup(well1: pl.DataFrame, well2: pl.DataFrame, key: jax.random.PRNGKey) -> pl.DataFrame:
    """Linearly interpolates between two wells.
    
    Args:
        well1: Polars DataFrame.
        well2: Polars DataFrame.
        key: JAX PRNG key.

    Returns:
        Augmented Polars DataFrame.
    """
    # Assuming same schema/length for simplicity.
    # If not, would need interpolation/resampling.
    length = min(len(well1), len(well2))
    
    # Cast to float, fill nulls before converting to numpy
    w1_df = well1.slice(0, length).select([pl.col(c).cast(pl.Float64).fill_null(0.0) for c in well1.columns])
    w2_df = well2.slice(0, length).select([pl.col(c).cast(pl.Float64).fill_null(0.0) for c in well2.columns])
    
    w1 = jnp.array(w1_df.to_numpy())
    w2 = jnp.array(w2_df.to_numpy())
    
    alpha = jax.random.uniform(key, (), minval=0.4, maxval=0.6)
    
    augmented_data = alpha * w1 + (1 - alpha) * w2
    
    # Convert JAX array back to numpy array
    return pl.DataFrame(np.array(augmented_data), schema=well1.columns)

def augment_all():
    """Main function to augment the dataset."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Get all horizontal_well csvs
    well_files = list(INPUT_DIR.glob('*__horizontal_well.csv'))
    
    # Need to keep track of all wells for mixup
    all_wells = []
    for f in well_files:
        all_wells.append(pl.read_csv(f))
        
    num_wells = len(well_files)
    
    key = jax.random.PRNGKey(42)
    
    for i, file_path in enumerate(well_files):
        # Original
        well_df = pl.read_csv(file_path)
        well_name = file_path.name
        
        # Save original
        well_df.write_csv(OUTPUT_DIR / well_name)
        
        # Augment
        key, subkey1, subkey2 = jax.random.split(key, 3)
        
        # Pick technique
        technique_idx = jax.random.randint(subkey1, (), 0, 6)
        
        if technique_idx == 0:
            aug_df = sensor_noise_jitter(well_df, subkey2)
        elif technique_idx == 1:
            aug_df = gaussian_scaling(well_df, subkey2)
        elif technique_idx == 2:
            aug_df = magnitude_warping(well_df, subkey2)
        elif technique_idx == 3:
            aug_df = trajectory_perturbation(well_df, subkey2)
        elif technique_idx == 4:
            aug_df = window_slicing(well_df, subkey2)
        else: # Mixup
            # Pick another random well
            other_idx = jax.random.randint(subkey2, (), 0, num_wells)
            aug_df = mixup(well_df, all_wells[other_idx], subkey2)
            
        # Save augmented
        aug_name = well_name.replace('.csv', '_aug.csv')
        aug_df.write_csv(OUTPUT_DIR / aug_name)
        print(f"Processed {well_name}, augmented with technique {technique_idx}")

if __name__ == "__main__":
    augment_all()
