"""Exploratory Data Analysis for well logs.

This module provides functions to visualize Gamma Ray logs and their relationship
to geological depth.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import polars as pl
import os
import sys

# Add src to path
sys.path.append(os.path.join(os.getcwd(), "src"))
from data_loader import load_well_pair, preprocess_logs, get_all_well_ids

def plot_well_logs(horiz_df, type_df, well_id, save_path):
    """Plots Gamma Ray logs for horizontal and type wells.
    
    Args:
        horiz_df: Horizontal well DataFrame.
        type_df: Typewell DataFrame.
        well_id: ID of the well.
        save_path: Directory to save the plot.
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10), sharex=False)
    
    # Plot Horizontal Well GR
    sns.lineplot(data=horiz_df.to_pandas(), x="MD", y="GR", ax=ax1, color="blue")
    ax1.set_title(f"Horizontal Well {well_id} - GR vs MD")
    ax1.set_ylabel("Gamma Ray (API)")
    
    # Plot Typewell GR
    sns.lineplot(data=type_df.to_pandas(), x="TVT", y="GR", ax=ax2, color="green")
    ax2.set_title(f"Typewell {well_id} - GR vs TVT")
    ax2.set_ylabel("Gamma Ray (API)")
    ax2.set_xlabel("True Vertical Thickness (ft)")
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, f"{well_id}_logs.png"))
    plt.close()

if __name__ == "__main__":
    DATA_PATH = "/home/samer/Documents/competitions/ROGII/dataset/"
    ANALYTICS_PATH = "/home/samer/Documents/competitions/ROGII/analytics/"
    
    well_ids = get_all_well_ids(DATA_PATH)
    
    # Sample 3 wells for EDA
    for well_id in well_ids[:3]:
        print(f"Generating EDA for well {well_id}...")
        h_df, t_df = load_well_pair(DATA_PATH, well_id)
        h_df = preprocess_logs(h_df)
        t_df = preprocess_logs(t_df)
        
        plot_well_logs(h_df, t_df, well_id, ANALYTICS_PATH)
    
    print(f"EDA plots saved to {ANALYTICS_PATH}")
