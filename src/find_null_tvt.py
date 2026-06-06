import polars as pl
import glob

def find_null_tvt():
    files = glob.glob('dataset/train/*__horizontal_well.csv')
    print(f"Scanning {len(files)} horizontal wells...")
    tvt_nulls = 0
    tvt_nans = 0
    tvt_input_nulls = 0
    tvt_input_nans = 0
    total_rows = 0
    
    for f in files:
        df = pl.read_csv(f, infer_schema_length=10000)
        n_tvt = df['TVT'].is_null().sum()
        # n_tvt_nan = df['TVT'].is_nan().sum() # is_nan only for floats
        
        tvt_nulls += n_tvt
        # tvt_nans += n_tvt_nan
        tvt_input_nulls += df['TVT_input'].is_null().sum()
        total_rows += len(df)
            
    print(f"TVT: {tvt_nulls} nulls out of {total_rows} rows")
    print(f"TVT_input: {tvt_input_nulls} nulls out of {total_rows} rows")

    files = glob.glob('dataset/train/*__typewell.csv')
    print(f"\nScanning {len(files)} typewells...")
    found = False
    for f in files:
        df = pl.read_csv(f, infer_schema_length=10000)
        n = df['TVT'].is_null().sum()
        if n > 0:
            print(f"{f}: {n} nulls")
            found = True
    if not found:
        print("No nulls found in typewells.")

if __name__ == "__main__":
    find_null_tvt()
