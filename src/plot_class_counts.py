"""Visualization script for class counts in the ROGII Wellbore Geology dataset.

This script aggregates and counts the occurrences of the target geological classes
(ANCC, ASTNU, ASTNL, EGFDU, EGFDL, BUDA) in both the vertical reference Type Wells
(stratigraphic thickness counts) and the horizontal well trajectories (drilled
exposure counts). It then generates a high-quality, professional dark-themed
visualization and saves it to the analytics directory.
"""

import os
import glob
import polars as pl
import matplotlib.pyplot as plt
import seaborn as sns

def count_typewell_classes(dataset_path: str, target_classes: list[str]) -> pl.DataFrame:
    """Aggregates class counts from the 'Geology' column across all Type Wells.

    Args:
        dataset_path: Path to the dataset root folder containing the train set.
        target_classes: List of stratigraphic classes to filter and count.

    Returns:
        A Polars DataFrame containing the class names and their respective counts,
        sorted in stratigraphic order.
    """
    train_path = os.path.join(dataset_path, "train")
    typewell_pattern = os.path.join(train_path, "*__typewell.csv")
    typewell_files = glob.glob(typewell_pattern)
    
    print(f"Scanning {len(typewell_files)} Type Well files...")
    
    # Use lazy frame list for highly efficient parallel loading
    lazy_frames = []
    for filepath in typewell_files:
        lf = pl.scan_csv(filepath, infer_schema_length=10000)
        # Only select Geology column if it exists to avoid errors
        lf = lf.select(["Geology"])
        lazy_frames.append(lf)
        
    # Concatenate all lazy frames and execute aggregation
    combined_df = pl.concat(lazy_frames).collect()
    
    # Filter non-null and strip whitespaces
    geology_counts = (
        combined_df
        .filter(pl.col("Geology").is_not_null())
        .with_columns(pl.col("Geology").str.strip_chars().alias("Geology"))
        .group_by("Geology")
        .len()
        .rename({"len": "Count"})
    )
    
    # Filter for the target classes and sort them in stratigraphic order
    ordered_counts = (
        pl.DataFrame({"Geology": target_classes})
        .join(geology_counts, on="Geology", how="left")
        .fill_null(0)
    )
    
    return ordered_counts

def count_horizontal_classes(dataset_path: str, target_classes: list[str]) -> pl.DataFrame:
    """Classifies wellbore Z coordinates relative to markers in horizontal wells.

    Args:
        dataset_path: Path to the dataset root folder containing the train set.
        target_classes: List of stratigraphic classes in top-down order.

    Returns:
        A Polars DataFrame containing the class names and their respective counts in
        the horizontal well trajectories.
    """
    train_path = os.path.join(dataset_path, "train")
    horizontal_pattern = os.path.join(train_path, "*__horizontal_well.csv")
    horiz_files = glob.glob(horizontal_pattern)
    
    print(f"Scanning {len(horiz_files)} Horizontal Well files...")
    
    # Define columns needed for boundary calculation
    cols_to_select = ["Z"] + target_classes
    
    lazy_frames = []
    for filepath in horiz_files:
        lf = pl.scan_csv(filepath, infer_schema_length=10000)
        # Cast any potentially parsed string columns to Float64
        lf = lf.select(cols_to_select).cast({pl.String: pl.Float64}, strict=False)
        lazy_frames.append(lf)
        
    combined_df = pl.concat(lazy_frames).collect()
    
    # Perform geometric layer classification
    # Markers represent boundaries in decreasing elevation order (more negative = deeper).
    # - ANCC layer: ASTNU < Z <= ANCC
    # - ASTNU layer: ASTNL < Z <= ASTNU
    # - ASTNL layer: EGFDU < Z <= ASTNL
    # - EGFDU layer: EGFDL < Z <= EGFDU
    # - EGFDL layer: BUDA < Z <= EGFDL
    # - BUDA layer: Z <= BUDA
    
    z = combined_df["Z"]
    m_ancc = combined_df["ANCC"]
    m_astnu = combined_df["ASTNU"]
    m_astnl = combined_df["ASTNL"]
    m_egfdu = combined_df["EGFDU"]
    m_egfdl = combined_df["EGFDL"]
    m_buda = combined_df["BUDA"]
    
    counts = {
        "ANCC": ((z <= m_ancc) & (z > m_astnu)).sum(),
        "ASTNU": ((z <= m_astnu) & (z > m_astnl)).sum(),
        "ASTNL": ((z <= m_astnl) & (z > m_egfdu)).sum(),
        "EGFDU": ((z <= m_egfdu) & (z > m_egfdl)).sum(),
        "EGFDL": ((z <= m_egfdl) & (z > m_buda)).sum(),
        "BUDA": ((z <= m_buda)).sum(),
    }
    
    # Clean up null values (in case some rows had nulls in Z or markers)
    counts = {k: int(v) if v is not None else 0 for k, v in counts.items()}
    
    ordered_counts = pl.DataFrame({
        "Geology": target_classes,
        "Count": [counts[c] for c in target_classes]
    })
    
    return ordered_counts

def plot_beautiful_barchart(typewell_df: pl.DataFrame, horiz_df: pl.DataFrame, save_path: str):
    """Generates an extremely professional, modern, and beautiful dual bar chart.

    Args:
        typewell_df: Class counts in vertical reference Type Wells.
        horiz_df: Class counts in horizontal well trajectories.
        save_path: File path to save the generated image.
    """
    # Convert Polars to Pandas for plotting compatibility
    tw_pd = typewell_df.to_pandas()
    hz_pd = horiz_df.to_pandas()
    
    # Set style to dark theme with custom styling parameters
    plt.style.use('dark_background')
    
    # Create side-by-side figures
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 8), facecolor='#0F172A')
    
    # Colors for the target formations: soft, cohesive pastel/neon gradient
    colors = ['#38BDF8', '#6366F1', '#EC4899', '#F43F5E', '#10B981', '#F59E0B']
    
    # Clean up grid and axes background colors for both axes
    for ax in [ax1, ax2]:
        ax.set_facecolor('#1E293B')
        ax.grid(True, linestyle='--', alpha=0.15, color='#FFFFFF')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_color('#475569')
        ax.spines['bottom'].set_color('#475569')
        ax.tick_params(colors='#94A3B8', labelsize=11)
        
    # Plot 1: Type Wells Distribution
    bars1 = ax1.bar(tw_pd["Geology"], tw_pd["Count"], color=colors, edgecolor='#0F172A', linewidth=1.5, alpha=0.9, width=0.6)
    ax1.set_title("Vertical Stratigraphic Thickness (Type Wells)\n[Labeled Reference Columns]", fontsize=14, fontweight='bold', pad=15, color='#F8FAFC')
    ax1.set_ylabel("Data Points Count", fontsize=12, fontweight='semibold', labelpad=10, color='#CBD5E1')
    ax1.set_xlabel("Geological Formations", fontsize=12, fontweight='semibold', labelpad=10, color='#CBD5E1')
    
    # Add values on top of bars
    for bar in bars1:
        yval = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2.0, yval + (yval * 0.015), f"{yval:,}", ha='center', va='bottom', fontsize=10, fontweight='bold', color='#E2E8F0')
        
    # Plot 2: Horizontal Wells Distribution
    bars2 = ax2.bar(hz_pd["Geology"], hz_pd["Count"], color=colors, edgecolor='#0F172A', linewidth=1.5, alpha=0.9, width=0.6)
    ax2.set_title("Drilled Trajectory Exposure (Horizontal Wells)\n[Time spent by drill bit inside each zone]", fontsize=14, fontweight='bold', pad=15, color='#F8FAFC')
    ax2.set_ylabel("Data Points Count", fontsize=12, fontweight='semibold', labelpad=10, color='#CBD5E1')
    ax2.set_xlabel("Geological Formations", fontsize=12, fontweight='semibold', labelpad=10, color='#CBD5E1')
    
    # Add values on top of bars
    for bar in bars2:
        yval = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2.0, yval + (yval * 0.015), f"{yval:,}", ha='center', va='bottom', fontsize=10, fontweight='bold', color='#E2E8F0')
        
    # Add main high-quality title and subheadings
    plt.suptitle("ROGII Geosteering Dataset - Formation Class Distributions", fontsize=18, fontweight='bold', y=0.98, color='#F8FAFC')
    plt.figtext(0.5, 0.02, "Note: EGFDL is the primary target zone for lateral geosteering placement.", ha="center", fontsize=11, style="italic", color='#94A3B8')
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    # Save the output image with high resolution
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='#0F172A')
    plt.close()
    print(f"Beautiful dual-bar chart saved to: {save_path}")

def main():
    """Main execution block to load data, count classes, and plot distributions."""
    dataset_path = "/home/samer/Documents/competitions/ROGII/dataset/"
    analytics_path = "/home/samer/Documents/competitions/ROGII/analytics/"
    os.makedirs(analytics_path, exist_ok=True)
    
    # The target formations/classes listed by the user
    target_classes = ["ANCC", "ASTNU", "ASTNL", "EGFDU", "EGFDL", "BUDA"]
    
    # 1. Count classes in Vertical reference Type Wells
    tw_counts = count_typewell_classes(dataset_path, target_classes)
    print("\n--- Type Well Stratigraphic Counts ---")
    print(tw_counts)
    
    # 2. Count class intervals in Horizontal Drilling wellbores
    hz_counts = count_horizontal_classes(dataset_path, target_classes)
    print("\n--- Horizontal Well Drilled Exposure Counts ---")
    print(hz_counts)
    
    # 3. Plot and save
    chart_output_path = os.path.join(analytics_path, "class_distributions.png")
    plot_beautiful_barchart(tw_counts, hz_counts, chart_output_path)

if __name__ == "__main__":
    main()
