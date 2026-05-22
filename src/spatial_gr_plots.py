"""Spatial and Gamma Ray visualization for ROGII Wellbore Geology dataset.

This script generates scatter plots for Gamma Ray (GR) values and spatial
coordinates (X, Y, Z) from horizontal well trajectories. It uses Polars for
efficient data handling and Matplotlib/Seaborn for high-quality visualization.
"""

import os
import glob
import polars as pl
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns
import jax.numpy as jnp

def load_spatial_gr_data(dataset_path: str, sample_fraction: float = 0.1) -> pl.DataFrame:
    """Loads spatial coordinates and GR values from horizontal wells.

    Args:
        dataset_path: Path to the dataset root folder.
        sample_fraction: Fraction of data to sample for visualization efficiency.

    Returns:
        A Polars DataFrame containing X, Y, Z, and GR columns.
    """
    train_path = os.path.join(dataset_path, "train")
    horizontal_pattern = os.path.join(train_path, "*__horizontal_well.csv")
    horiz_files = glob.glob(horizontal_pattern)
    
    print(f"Loading spatial and GR data from {len(horiz_files)} files...")
    
    cols_to_select = ["X", "Y", "Z", "GR"]
    
    lazy_frames = []
    for filepath in horiz_files:
        lf = pl.scan_csv(filepath, infer_schema_length=10000)
        # Select and cast to Float64
        lf = lf.select(cols_to_select).cast({pl.String: pl.Float64}, strict=False)
        lazy_frames.append(lf)
        
    combined_df = pl.concat(lazy_frames).collect()
    
    # Drop rows with null values in required columns
    combined_df = combined_df.drop_nulls(subset=cols_to_select)
    
    # Sample data if it's too large for responsive plotting
    if sample_fraction < 1.0:
        combined_df = combined_df.sample(fraction=sample_fraction, seed=42)
        
    return combined_df

def plot_gr_scatter(df: pl.DataFrame, save_path: str):
    """Generates a scatter plot for Gamma Ray (GR) values vs Vertical Depth (Z).

    Args:
        df: Polars DataFrame with 'Z' and 'GR' columns.
        save_path: File path to save the generated image.
    """
    plt.style.use('dark_background')
    fig, ax = plt.subplots(figsize=(12, 8), facecolor='#0F172A')
    
    ax.set_facecolor('#1E293B')
    ax.grid(True, linestyle='--', alpha=0.15, color='#FFFFFF')
    
    # Using a scatter plot as requested
    sns.scatterplot(
        x=df["GR"].to_numpy(),
        y=df["Z"].to_numpy(),
        alpha=0.4,
        s=10,
        color='#38BDF8',
        edgecolor=None,
        ax=ax
    )
    
    ax.set_title("Gamma Ray (GR) Distribution vs Vertical Depth (Z)", fontsize=16, fontweight='bold', pad=20, color='#F8FAFC')
    ax.set_xlabel("Gamma Ray (API)", fontsize=12, fontweight='semibold', color='#CBD5E1')
    ax.set_ylabel("Vertical Depth (Z)", fontsize=12, fontweight='semibold', color='#CBD5E1')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='#0F172A')
    plt.close()
    print(f"GR scatter plot saved to: {save_path}")

def plot_spatial_3d_scatter(df: pl.DataFrame, save_path: str):
    """Generates a 3D scatter plot for spatial coordinates (X, Y, Z).

    Args:
        df: Polars DataFrame with 'X', 'Y', and 'Z' columns.
        save_path: File path to save the generated image.
    """
    plt.style.use('dark_background')
    fig = plt.figure(figsize=(12, 10), facecolor='#0F172A')
    ax = fig.add_subplot(111, projection='3d', facecolor='#0F172A')
    
    ax.set_facecolor('#0F172A')
    
    # Scatter plot in 3D
    scatter = ax.scatter(
        df["X"].to_numpy(),
        df["Y"].to_numpy(),
        df["Z"].to_numpy(),
        c=df["Z"].to_numpy(),
        cmap='viridis',
        alpha=0.5,
        s=5
    )
    
    ax.set_title("3D Spatial Wellbore Trajectories (X, Y, Z)", fontsize=16, fontweight='bold', pad=20, color='#F8FAFC')
    ax.set_xlabel("X Coordinate", fontsize=12, fontweight='semibold', color='#CBD5E1', labelpad=10)
    ax.set_ylabel("Y Coordinate", fontsize=12, fontweight='semibold', color='#CBD5E1', labelpad=10)
    ax.set_zlabel("Z Coordinate (Depth)", fontsize=12, fontweight='semibold', color='#CBD5E1', labelpad=10)
    
    # Color bar for depth
    cbar = plt.colorbar(scatter, ax=ax, pad=0.1, shrink=0.7)
    cbar.set_label('Depth (Z)', color='#CBD5E1')
    cbar.ax.yaxis.set_tick_params(color='#CBD5E1', labelcolor='#CBD5E1')
    
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='#0F172A')
    plt.close()
    print(f"Spatial 3D scatter plot saved to: {save_path}")

def main():
    """Main execution block."""
    dataset_path = "/home/samer/Documents/competitions/ROGII/dataset/"
    analytics_path = "/home/samer/Documents/competitions/ROGII/analytics/"
    os.makedirs(analytics_path, exist_ok=True)
    
    # Load and sample data (10% sample for performance)
    df = load_spatial_gr_data(dataset_path, sample_fraction=0.05)
    
    # 1. GR vs Z Scatter Plot
    gr_plot_path = os.path.join(analytics_path, "gr_scatter.png")
    plot_gr_scatter(df, gr_plot_path)
    
    # 2. 3D Spatial Scatter Plot
    spatial_plot_path = os.path.join(analytics_path, "spatial_3d_scatter.png")
    plot_spatial_3d_scatter(df, spatial_plot_path)

if __name__ == "__main__":
    main()
