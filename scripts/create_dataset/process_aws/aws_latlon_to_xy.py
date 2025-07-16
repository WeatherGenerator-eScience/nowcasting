import os
import glob
import pyproj
import pandas as pd
import numpy as np
from tqdm import tqdm

# Constants
START_TIME = pd.Timestamp("2020-01-01 00:00")
END_TIME = pd.Timestamp("2023-12-31 23:59")
INTERVAL_MINUTES = 10

class RadarGridConverter:
    def __init__(self):
        # Original grid configuration (match your radar dimensions)
        self.original_cols = 700
        self.original_rows = 765
        self.pad_to_divisible = 32  # Must match AwsImageGenerator's pad_to_divisible
        
        # Compute padded dimensions to be divisible by pad_to_divisible
        self.grid_cols = ((self.original_cols + self.pad_to_divisible - 1) // self.pad_to_divisible) * self.pad_to_divisible
        self.grid_rows = ((self.original_rows + self.pad_to_divisible - 1) // self.pad_to_divisible) * self.pad_to_divisible
        
        # Create projection system (must match AwsImageGenerator)
        self.proj = pyproj.Proj(
            "+proj=stere +lat_0=90 +lon_0=0 +lat_ts=60 "
            "+a=6378.14 +b=6356.75 +x_0=0 +y_0=0"
        )
        
        # Calculate original grid extents (same as AwsImageGenerator)
        self.x_ul, self.y_ul = self.proj(0, 55.973602)  # Upper left (UL corner)
        x_lr_original, y_lr_original = self.proj(9.0093, 48.8953)  # Original lower right (LR corner)
        
        # Calculate grid step sizes from original grid
        self.dx = (x_lr_original - self.x_ul) / (self.original_cols - 1)
        self.dy = (y_lr_original - self.y_ul) / (self.original_rows - 1)
        
        # Calculate extended grid extents for padded grid
        self.x_lr_padded = self.x_ul + self.dx * (self.grid_cols - 1)
        self.y_lr_padded = self.y_ul + self.dy * (self.grid_rows - 1)

    def latlon_to_grid_indices(self, df):
        """Convert geographic coordinates to padded grid indices (uint16)"""
        x, y = self.proj(df['DS_LON'].values, df['DS_LAT'].values)
        
        # Convert to grid indices using padded extents
        cols = np.clip(
            np.round((x - self.x_ul) / (self.x_lr_padded - self.x_ul) * (self.grid_cols - 1)),
            0, self.grid_cols - 1
        ).astype(np.uint16)
        
        rows = np.clip(
            np.round((y - self.y_ul) / (self.y_lr_padded - self.y_ul) * (self.grid_rows - 1)),
            0, self.grid_rows - 1
        ).astype(np.uint16)
        
        return cols, rows

def create_time_index_map():
    """Create timestamp -> index mapping for 10-minute intervals"""
    timestamps = pd.date_range(
        start=START_TIME,
        end=END_TIME,
        freq=f'{INTERVAL_MINUTES}min'
    )
    return {ts: idx for idx, ts in enumerate(timestamps)}

def process_file(input_path, output_dir, converter, time_map):
    """Process and convert a single Parquet file"""
    try:
        df = pd.read_parquet(input_path)
        
        # Convert timestamps to time indices
        df['time_index'] = df['IT_DATETIME'].map(time_map)
        
        # Filter invalid timestamps
        valid_mask = df['time_index'].notna()
        if not valid_mask.any():
            print(f"Skipping {os.path.basename(input_path)} - no valid timestamps")
            return False
            
        df = df[valid_mask].copy()
        
        # Convert coordinates
        cols, rows = converter.latlon_to_grid_indices(df)
        
        # Create simplified dataframe
        converted_df = pd.DataFrame({
            'X': cols,
            'Y': rows,
            'T': df['time_index'].astype(np.uint32),
            'value': df['value']
        }).reset_index(drop=True)
        
        # Save results
        output_path = os.path.join(output_dir, os.path.basename(input_path))
        converted_df.to_parquet(output_path)
        return True
        
    except Exception as e:
        print(f"Error processing {os.path.basename(input_path)}: {str(e)}")
        return False

def main():
    input_dir = "/nobackup_1/users/robben/full_dataset/aws/latlon"
    output_dir = "/nobackup_1/users/robben/full_dataset/aws/xy"
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize components
    converter = RadarGridConverter()
    time_map = create_time_index_map()
    parquet_files = glob.glob(os.path.join(input_dir, "*.parquet"))
    
    print(f"Time index range: 0-{len(time_map)-1}")
    print(f"Processing {len(parquet_files)} files...")
    
    # Process files with progress bar
    success_count = 0
    for file_path in tqdm(parquet_files):
        if process_file(file_path, output_dir, converter, time_map):
            success_count += 1
            
    print(f"\nConversion complete. Success rate: {success_count}/{len(parquet_files)}")
    print(f"Output format: X (uint16), Y (uint16), T (uint32), value")

if __name__ == "__main__":
    main()