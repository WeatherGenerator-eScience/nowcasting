import zarr
import numpy as np
import pandas as pd
import dask.array as da

from typing import Dict, List, Any, Optional

from nowcasting.data.utils.generate_samples import assign_splits

print('Start to compute statistics of weather data.')

# --- Setup for Zarr Data Access and Processing Configuration ---
# Defines the rules for splitting the dataset into training, validation, and testing periods.
# - "test": All data from the year 2023.
# - "val": All data from months June (6) and November (11), excluding any data already
#          assigned to "test" (i.e., June/Nov 2023 will be in 'test').
# - "train": All remaining data not assigned to "test" or "val".
split_rules = {
    "test": {"year": [2023]},
    "val":  {"month": [6, 11]},
    "train": {} # Empty dictionary for 'train' means it's the default catch-all.
}

# Path to the root Zarr store. 
path = '/vol/knmimo-nobackup/users/mrobben/nowcasting/data/dataset_regrid.zarr'

# Open the Zarr root group in read-only mode.
root = zarr.open_group(path, mode='r')

# Configuration for each Zarr group to be processed.
# 'excluded' lists variables within each group that should *not* be processed
# for statistics (e.g., metadata like lat/lon, or masks).
groups = {
    'radar': {'excluded': ['lat', 'lon', 'max_intensity_grid', 'time_mask']},
    'harmonie': {'excluded': ['lat', 'lon', 'time_mask']},
    'sat_l2': {'excluded': ['lat', 'lon', 'time_mask']},
    'sat_l1p5': {'excluded': ['lat', 'lon', 'time_mask']}
}

results = {}

# --- Dask Processing Loop ---
for group_name, cfg in groups.items():
    group = root[group_name]

    start_time, end_time, interval_minutes = group.attrs['start_time'], group.attrs['end_time'], group.attrs['interval_minutes']

    # Generate a full DatetimeIndex for the current group's time axis
    full_times = pd.date_range(start_time, end_time, freq=f"{interval_minutes}min")

    # Assign time indices to 'train', 'val', 'test' splits using the `split_rules`.
    splits = assign_splits(full_times, split_rules)

    # Load the time mask for the current group
    time_mask = group['time_mask'][:]

    # Determine the valid training indices by intersecting the 'train' split indices
    # with the indices where `time_mask` is True.
    valid_indices = np.intersect1d(splits['train'], np.nonzero(time_mask)[0], assume_unique=True)

    results[group_name] = {'means': {}, 'stds': {}, 'mean_images': {}, 'std_images': {}}
    
    print(f"\n--- Processing Group: {group_name} with Dask ---")

    # Iterates through each configured group (e.g., 'radar', 'harmonie') to compute statistics.
    for var_name in group.array_keys():
        if var_name in cfg['excluded']:
            continue

        arr = group[var_name]
        print(f"Processing {var_name} with shape {arr.shape}")

        # If there's no valid training data for this variable, skip.
        if len(valid_indices) == 0:
            print(f"  → No valid data found for {var_name}.")
            continue
        
        # Create a Dask array from the Zarr array. This allows for lazy, out-of-core computation.
        dask_arr = da.from_zarr(arr)
        train_data = dask_arr[valid_indices, :, :].astype(np.float64)

        # Define Dask computations for overall sum, sum of squares, and count.
        total_sum = train_data.sum()
        total_sum_sq = da.square(train_data).sum()
        total_count = train_data.size

        total_min = train_data.min()
        total_max = train_data.max()
        total_p5, total_p95 = da.percentile(train_data.flatten(), q=[5, 95]) 

        print("  → Starting parallel computation with Dask...")
        total_sum_res, total_sum_sq_res, total_min_res, total_max_res, total_p5_res, total_p95_res = da.compute(
            total_sum, total_sum_sq, total_min, total_max, total_p5, total_p95
        )
        print("  → Computation finished.")

        mean = total_sum_res / total_count
        std = np.sqrt(total_sum_sq_res / total_count - mean**2)
        
        results[group_name]['means'][var_name] = mean
        results[group_name]['stds'][var_name] = std
        results[group_name]['mins'][var_name] = total_min_res
        results[group_name]['maxs'][var_name] = total_max_res
        results[group_name]['p5s'][var_name] = total_p5_res
        results[group_name]['p95s'][var_name] = total_p95_res
        print(f"  → mean: {mean:.4f}, std: {std:.4f}")
        print(f"  → min: {total_min_res:.4f}, max: {total_max_res:.4f}")
        print(f"  → p5: {total_p5_res:.4f}, p95: {total_p95_res:.4f}")

# ===============================================================
# --- ADDED SECTION: Saving the results to a Zarr file ---
# ===============================================================
print("\n--- Saving all results to computation_results.zarr ---")
results_store = zarr.open_group('/vol/knmimo-nobackup/users/mrobben/nowcasting/data/stats.zarr', mode='a')

for group_name, group_data in results.items():
    # Create a top-level group for 'radar', 'sat_l2', etc.
    res_group = results_store.create_group(group_name)

    # Save the scalar means and stds as dictionary attributes of the group
    res_group.attrs['means'] = group_data['means']
    res_group.attrs['stds'] = group_data['stds']
    res_group.attrs['mins'] = group_data['mins']
    res_group.attrs['maxs'] = group_data['maxs']
    res_group.attrs['p5s'] = group_data['p5s']
    res_group.attrs['p95s'] = group_data['p95s']

print("Results successfully saved.")