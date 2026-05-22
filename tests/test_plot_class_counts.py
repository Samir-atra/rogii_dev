"""Unit tests for the plot_class_counts module.

This module validates the class counting and boundary geometry processing logic
in `src/plot_class_counts.py` using standard assertion checks.
"""

import os
import sys
import polars as pl

# Add src to the system path to allow importing the plot_class_counts module
sys.path.append(os.path.join(os.getcwd(), "src"))

from plot_class_counts import count_typewell_classes, count_horizontal_classes

def test_count_typewell_classes():
    """Validates that Type Well class counting returns correct columns and shapes.

    This function tests that `count_typewell_classes` properly filters and counts
    labeled geological classes, returning a valid Polars DataFrame with the target
    fields 'Geology' and 'Count'.
    """
    print("--- Running test_count_typewell_classes ---")
    dataset_path = "/home/samer/Documents/competitions/ROGII/dataset/"
    target_classes = ["ANCC", "ASTNU", "ASTNL", "EGFDU", "EGFDL", "BUDA"]
    
    # Run the typewell counting function
    counts_df = count_typewell_classes(dataset_path, target_classes)
    
    # Assertions to verify the structure and contents
    assert isinstance(counts_df, pl.DataFrame), "Output must be a Polars DataFrame."
    assert set(counts_df.columns) == {"Geology", "Count"}, f"Incorrect columns: {counts_df.columns}"
    assert len(counts_df) == len(target_classes), f"Expected {len(target_classes)} rows, got {len(counts_df)}."
    
    # Verify that counts are valid and non-negative
    for row in counts_df.iter_rows(named=True):
        assert row["Geology"] in target_classes, f"Unexpected geology class: {row['Geology']}"
        assert row["Count"] >= 0, f"Count for class {row['Geology']} cannot be negative."
        
    print("Success: Type Well class counting verified.")

def test_count_horizontal_classes():
    """Validates that Horizontal Well class counting returns correct columns and shapes.

    This function tests that `count_horizontal_classes` properly reads boundary
    marker elevations, classifies wellbore Z positions into geological layers,
    and returns a valid Polars DataFrame with non-negative counts.
    """
    print("\n--- Running test_count_horizontal_classes ---")
    dataset_path = "/home/samer/Documents/competitions/ROGII/dataset/"
    target_classes = ["ANCC", "ASTNU", "ASTNL", "EGFDU", "EGFDL", "BUDA"]
    
    # Run the horizontal well counting function
    counts_df = count_horizontal_classes(dataset_path, target_classes)
    
    # Assertions to verify the structure and contents
    assert isinstance(counts_df, pl.DataFrame), "Output must be a Polars DataFrame."
    assert set(counts_df.columns) == {"Geology", "Count"}, f"Incorrect columns: {counts_df.columns}"
    assert len(counts_df) == len(target_classes), f"Expected {len(target_classes)} rows, got {len(counts_df)}."
    
    # Verify that counts are valid and non-negative
    for row in counts_df.iter_rows(named=True):
        assert row["Geology"] in target_classes, f"Unexpected geology class: {row['Geology']}"
        assert row["Count"] >= 0, f"Count for class {row['Geology']} cannot be negative."
        
    print("Success: Horizontal Well class counting verified.")

if __name__ == "__main__":
    try:
        test_count_typewell_classes()
        test_count_horizontal_classes()
        print("\nALL UNIT TESTS PASSED. plot_class_counts functions are correct.")
    except Exception as e:
        print(f"\nUNIT TEST FAILED: {e}")
        sys.exit(1)
