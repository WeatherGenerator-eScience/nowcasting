import os
import glob
import shutil
import pandas as pd
import numpy as np
from pathlib import Path
from fastparquet import write

# === User settings ===
DATA_DIR = Path("data/")                # directory containing your CSV files
OUTPUT_DIR = Path("output/")    # base directory where per-variable Parquet files will be created
TEST_MODE = False                         # set True to process only a subset for testing
TEST_FILES = 1                            # number of files to test in TEST_MODE
COMPRESSION = 'GZIP'                      # compression algorithm: GZIP, SNAPPY, etc.
ROW_GROUP_SIZE = 100_000                  # approximate rows per parquet row-group

# Prefix columns expected in every file
PREFIX_COLS = [
    "IT_DATETIME", "DS_CODE", "DS_NAME", "DS_DESC",
    "DS_LAT", "DS_LON", "DS_ALT"
]

def fix_datetime_format(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert invalid '2400' times to valid '0000' of next day in DataFrame.
    Returns DataFrame with proper datetime index.
    """
    # Split datetime string into components
    parts = df['IT_DATETIME'].str.split('_', expand=True)
    date_part = parts[0]
    time_part = parts[1]
    
    # Create mask for invalid times (240000)
    invalid_mask = time_part == '240000'
    
    # Convert dates to datetime and add 1 day where needed
    dates = pd.to_datetime(date_part, format='%Y%m%d', errors='coerce')
    dates += pd.to_timedelta(invalid_mask.astype(int), unit='D')
    
    # Format corrected components
    corrected_date = dates.dt.strftime('%Y%m%d')
    corrected_time = np.where(invalid_mask, '000000', time_part)
    
    # Rebuild datetime strings and parse
    df['IT_DATETIME'] = pd.to_datetime(
        corrected_date + '_' + corrected_time + '_' + parts[2],
        format='%Y%m%d_%H%M%S_%f'
    )
    
    return df

# Helpers to parse dates from filenames
def parse_filename_dates(fn: str):
    parts = Path(fn).stem.split("_")  # ..._YYYYMMDD_HHMMSS_YYYYMMDD_HHMMSS
    if len(parts) >= 6:
        return parts[-4], parts[-2]
    return None, None

# Sanitize variable names for filenames
def sanitize_varname(v: str) -> str:
    return v.replace('.', '_').replace(' ', '_')

# Gather CSV files
all_files = sorted(glob.glob(str(DATA_DIR / "*.csv")))
if TEST_MODE:
    all_files = all_files[:TEST_FILES]
if not all_files:
    raise RuntimeError(f"No CSV files found in {DATA_DIR}")

# Compute global time span
starts, ends = zip(*(parse_filename_dates(f) for f in all_files))
global_start, global_end = min(starts), max(ends)

# Ensure output base directory exists
if OUTPUT_DIR.exists():
    shutil.rmtree(OUTPUT_DIR)
OUTPUT_DIR.mkdir(parents=True, exist_ok=False)

# Read header of first file to identify measurement columns
sample = pd.read_csv(all_files[0], nrows=0)
cols0 = sample.columns.tolist()
if cols0[:len(PREFIX_COLS)] != PREFIX_COLS:
    print(f"WARNING: prefix columns differ in first file: {all_files[0]}")
meas_cols = cols0[len(PREFIX_COLS):]

# Process each CSV file
for fn in all_files:
    print(f"Processing {fn}...")
    # Load, skipping malformed lines
    df = pd.read_csv(fn, on_bad_lines='warn')

    # Filter out station located outside the Netherlands
    df = df[~df["DS_CODE"].str.contains('871|873|990', na=False)]

    # Fix timestamps
    df = fix_datetime_format(df)

    # Validate structure
    cols = df.columns.tolist()
    prefix, meas = cols[:len(PREFIX_COLS)], cols[len(PREFIX_COLS):]
    if prefix != PREFIX_COLS:
        print(f"! Prefix columns mismatch in {fn}: {prefix}")
    if set(meas) != set(meas_cols):
        added = set(meas) - set(meas_cols)
        removed = set(meas_cols) - set(meas)
        print(f"! Measurement columns differ in {fn}:")
        if added:   print(f"  + New cols: {sorted(added)}")
        if removed: print(f"  - Missing cols: {sorted(removed)}")

    # Loop per measurement variable
    for var in meas_cols:
        sub = df.loc[df[var].notna(), PREFIX_COLS + [var]].copy()
        if sub.empty:
            continue
        # Extract StationID and rename value column
        sub['StationID'] = sub['DS_CODE'].str[:3]
        sub = sub.rename(columns={var: 'value'})
        # Sort by timestamp then StationID
        sub = sub.sort_values(['IT_DATETIME', 'StationID'])
        # Select final columns
        out_cols = ['StationID', 'DS_LAT', 'DS_LON', 'IT_DATETIME', 'value']
        sub = sub[out_cols]

        # Cast to smaller dtypes
        sub['StationID'] = sub['StationID'].astype('uint16')       # 3-digit codes fit in uint16
        sub[['DS_LAT', 'DS_LON']] = sub[['DS_LAT', 'DS_LON']].astype('float16')
        sub['value'] = sub['value'].astype('float32')

        # Write into per-variable Parquet file with compression and row-group sizing
        var_name = sanitize_varname(var)
        out_file = OUTPUT_DIR / f"{var_name}.parquet"
        write_args = {
            'compression': COMPRESSION,
            'row_group_offsets': ROW_GROUP_SIZE
        }
        if out_file.exists():
            write(out_file, sub, append=True, **write_args)
        else:
            write(out_file, sub, **write_args)

print("All files processed into Parquet under", OUTPUT_DIR)
