"""Missing value analysis for the ROGII Wellbore Geology dataset.

This script scans all CSV files in the dataset (train and test) to count
NaN/null values per feature using Polars for high-performance aggregation.
"""

import os
import glob
import polars as pl

def analyze_missing_values(dataset_path: str) -> pl.DataFrame:
    """Scans all CSV files and aggregates null counts per column.

    Args:
        dataset_path: Root directory of the dataset.

    Returns:
        A Polars DataFrame with columns 'Feature', 'Null_Count', and 'Percentage'.
    """
    # Find all CSV files in train and test
    csv_files = glob.glob(os.path.join(dataset_path, "**", "*.csv"), recursive=True)
    print(f"Analyzing {len(csv_files)} files for missing values...")

    # We use a dictionary to accumulate counts across all files
    total_counts = {}
    total_rows = 0

    for filepath in csv_files:
        # Scan the file lazily
        lf = pl.scan_csv(filepath, infer_schema_length=10000)
        
        # Get schema to iterate columns
        schema = lf.schema
        
        # Calculate nulls per column and total row count for this file
        # We use a list of expressions to avoid name collisions
        null_exprs = [pl.col(c).is_null().sum().alias(f"null_{c}") for c in schema.keys()]
        df_stats = lf.select([
            *null_exprs,
            pl.len().alias("row_count")
        ]).collect()
        
        file_rows = df_stats["row_count"][0]
        total_rows += file_rows
        
        for col in schema.keys():
            count = df_stats[f"null_{col}"][0]
            total_counts[col] = total_counts.get(col, 0) + count

    # Convert results to a DataFrame
    features = list(total_counts.keys())
    counts = [total_counts[f] for f in features]
    
    result_df = pl.DataFrame({
        "Feature": features,
        "Null_Count": counts
    }).with_columns(
        (pl.col("Null_Count") / total_rows * 100).round(2).alias("Percentage (%)")
    ).sort("Null_Count", descending=True)

    return result_df

def main():
    """Main execution block."""
    dataset_path = "/home/samer/Documents/competitions/ROGII/dataset/"
    
    null_report = analyze_missing_values(dataset_path)
    
    print("\n--- Missing Value Report ---")
    print(null_report)
    
    # Save the report for later use if needed
    analytics_path = "/home/samer/Documents/competitions/ROGII/analytics/"
    os.makedirs(analytics_path, exist_ok=True)
    null_report.write_csv(os.path.join(analytics_path, "missing_values_report.csv"))

if __name__ == "__main__":
    main()
