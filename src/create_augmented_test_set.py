"""Augment the test set by adding a TVT column.

This script creates a new directory that mirrors the original test set,
but with a key modification to the horizontal well files. It adds a 'TVT'
column and populates it with the values from the 'TVT_input' column.

This allows the labeled portions of the test set to be seamlessly integrated
into the training pipeline, which expects a 'TVT' target column.

Other files (e.g., typewell.csv, .png) are copied without modification.
"""

import os
import glob
import shutil
import polars as pl

def create_augmented_test_set(source_dir: str, dest_dir: str):
    """
    Creates a new test set directory where horizontal well files have a 'TVT' column
    copied from 'TVT_input'. Other files are copied as is.

    Args:
        source_dir: The path to the original 'test' directory.
        dest_dir: The path to the new directory for the augmented data.
    """
    if not os.path.exists(source_dir):
        print(f"Error: Source directory not found at '{source_dir}'")
        return

    os.makedirs(dest_dir, exist_ok=True)
    print(f"Created destination directory: '{dest_dir}'")

    files_to_process = glob.glob(os.path.join(source_dir, "*"))
    print(f"Found {len(files_to_process)} files to process in '{source_dir}'.")

    for file_path in files_to_process:
        file_name = os.path.basename(file_path)
        dest_path = os.path.join(dest_dir, file_name)

        if file_name.endswith("__horizontal_well.csv"):
            try:
                # Read, add the TVT column, and write back
                df = pl.read_csv(file_path, infer_schema_length=10000)
                df_with_tvt = df.with_columns(TVT=pl.col("TVT_input"))
                df_with_tvt.write_csv(dest_path)
                print(f"Processed and saved: {file_name}")
            except Exception as e:
                print(f"Error processing {file_name}: {e}")
        else:
            # For other files (typewell, png), just copy them
            shutil.copy(file_path, dest_path)
            print(f"Copied: {file_name}")

    print("\nAugmentation complete.")

def main():
    """Main execution block."""
    source_directory = "/home/samer/Documents/competitions/ROGII/dataset/test"
    destination_directory = "/home/samer/Documents/competitions/ROGII/dataset/test_with_tvt"
    create_augmented_test_set(source_directory, destination_directory)

if __name__ == "__main__":
    main()