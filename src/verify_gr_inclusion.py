"""Verify GR value inclusion between horizontal and type wells.

This script checks if the set of Gamma Ray (GR) values found in a
horizontal well file is fully contained within the set of GR values
in its corresponding vertical type well.

This analysis helps understand the relationship between the real-time
log signatures and the reference signatures.
"""

import os
import glob
import polars as pl

def get_test_well_ids(base_path: str) -> list[str]:
    """Retrieves all well IDs from the test dataset directory.

    Args:
        base_path: Root directory containing the 'test' folder.

    Returns:
        A list of unique well IDs from the test set.
    """
    test_path = os.path.join(base_path, "test")
    files = glob.glob(os.path.join(test_path, "*__horizontal_well.csv"))
    return sorted([os.path.basename(f).split("__")[0] for f in files])

def check_gr_inclusion(dataset_path: str):
    """
    Checks if horizontal well GR values are a subset of type well GR values.

    Args:
        dataset_path: The root path of the dataset.
    """
    well_ids = get_test_well_ids(dataset_path)
    test_dir = os.path.join(dataset_path, "test")

    if not well_ids:
        print("No test wells found.")
        return

    print(f"Analyzing {len(well_ids)} wells in '{test_dir}'...")

    fully_included_count = 0
    not_included_wells = []

    for well_id in well_ids:
        horiz_path = os.path.join(test_dir, f"{well_id}__horizontal_well.csv")
        typewell_path = os.path.join(test_dir, f"{well_id}__typewell.csv")

        horiz_gr = pl.read_csv(horiz_path).get_column("GR").drop_nulls().unique().to_list()
        typewell_gr = pl.read_csv(typewell_path).get_column("GR").drop_nulls().unique().to_list()

        horiz_gr_set = set(horiz_gr)
        typewell_gr_set = set(typewell_gr)

        if horiz_gr_set.issubset(typewell_gr_set):
            fully_included_count += 1
        else:
            not_included_wells.append(well_id)

    print("\n--- GR Inclusion Analysis Results ---")
    print(f"Total wells analyzed: {len(well_ids)}")
    print(f"Wells where horizontal GR is fully included in typewell GR: {fully_included_count}")
    print(f"Wells with GR values not found in typewell: {len(not_included_wells)}")

def main():
    """Main execution block."""
    dataset_path = "/home/samer/Documents/competitions/ROGII/dataset/"
    check_gr_inclusion(dataset_path)

if __name__ == "__main__":
    main()