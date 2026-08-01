"""Feature distribution visualization for the ROGII Wellbore Geology dataset.

This script generates scatter plots for each feature used in the training
notebook `08_conv1d_training.ipynb` to visualize their distributions and trends
across the entire training set.
"""

import os
import glob
import polars as pl
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

FEATURE_COLS = ["MD", "X", "Y", "Z", "GR", "TVT_input"]


def load_features_data(dataset_path: str, sample_fraction: float = 0.1) -> pl.DataFrame:
    """Loads all features from all training horizontal wells.

    Args:
        dataset_path: Path to the dataset root folder.
        sample_fraction: Fraction of data to sample for visualization efficiency.

    Returns:
        A Polars DataFrame containing the feature columns.
    """
    train_path = os.path.join(dataset_path, "train")
    horizontal_pattern = os.path.join(train_path, "*__horizontal_well.csv")
    horiz_files = glob.glob(horizontal_pattern)

    print(f"Loading feature data from {len(horiz_files)} training files...")

    lazy_frames = [
        pl.scan_csv(filepath, infer_schema_length=10000)
        .select(FEATURE_COLS)
        .cast({pl.String: pl.Float64}, strict=False)
        for filepath in horiz_files
    ]

    if not lazy_frames:
        return pl.DataFrame({col: [] for col in FEATURE_COLS})

    combined_df = pl.concat(lazy_frames).collect().drop_nulls()

    if sample_fraction < 1.0:
        print(f"Sampling {sample_fraction*100:.1f}% of the data for plotting.")
        combined_df = combined_df.sample(fraction=sample_fraction, seed=42)

    return combined_df


def plot_feature_distributions(df: pl.DataFrame, save_path: str):
    """Generates and saves scatter plots for each feature distribution.

    Args:
        df: Polars DataFrame with feature columns.
        save_path: File path to save the generated image.
    """
    if df.is_empty():
        print("DataFrame is empty, skipping plot generation.")
        return

    plt.style.use("dark_background")
    n_features = len(FEATURE_COLS)
    fig, axes = plt.subplots(
        n_features, 1, figsize=(12, 4 * n_features), facecolor="#0F172A"
    )

    if n_features == 1:
        axes = [axes]

    for i, col in enumerate(FEATURE_COLS):
        ax = axes[i]
        ax.set_facecolor("#1E293B")
        ax.grid(True, linestyle="--", alpha=0.15, color="#FFFFFF")

        values = df[col].to_numpy()
        indices = np.arange(len(values))

        sns.scatterplot(x=indices, y=values, ax=ax, s=5, alpha=0.5, edgecolor=None)

        ax.set_title(f"Distribution of {col}", fontsize=14, color="#F8FAFC")
        ax.set_xlabel("Sample Index", fontsize=10, color="#CBD5E1")
        ax.set_ylabel("Value", fontsize=10, color="#CBD5E1")

    plt.suptitle(
        "Scatter Plots of Training Feature Distributions",
        fontsize=18,
        fontweight="bold",
        y=1.0,
        color="#F8FAFC",
    )
    plt.tight_layout(rect=[0, 0, 1, 0.98])
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Feature distribution plots saved to: {save_path}")


if __name__ == "__main__":
    dataset_path = "/home/samer/Documents/competitions/ROGII/dataset/"
    analytics_path = "/home/samer/Documents/competitions/ROGII/analytics/"
    os.makedirs(analytics_path, exist_ok=True)

    features_df = load_features_data(dataset_path, sample_fraction=0.05)
    plot_path = os.path.join(analytics_path, "feature_distribution_scatter.png")
    plot_feature_distributions(features_df, plot_path)