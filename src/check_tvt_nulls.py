import polars as pl
import glob
import os

def check_tvt_nulls():
    train_files = glob.glob('dataset/train/*__horizontal_well.csv')
    test_files = glob.glob('dataset/test/*__horizontal_well.csv')
    
    print(f"Checking {len(train_files)} train files...")
    train_nulls = 0
    train_total = 0
    for f in train_files:
        df = pl.read_csv(f, infer_schema_length=10000)
        train_nulls += df["TVT"].is_null().sum()
        train_total += len(df)
    print(f"Train: {train_nulls} nulls out of {train_total} rows")

    print(f"\nChecking {len(test_files)} test files...")
    test_nulls = 0
    test_total = 0
    for f in test_files:
        df = pl.read_csv(f, infer_schema_length=10000)
        # Check if TVT column exists
        if "TVT" in df.columns:
            test_nulls += df["TVT"].is_null().sum()
            test_total += len(df)
        else:
            print(f"TVT column missing in {f}")
    print(f"Test: {test_nulls} nulls out of {test_total} rows")

if __name__ == "__main__":
    check_tvt_nulls()
