import zarr
import os
import glob
import pandas as pd
import numpy as np
import multiprocessing as mp
from functools import partial

BASE_DATA_DIR = "/nobackup_1/users/robben/full_dataset"
ZARR_OUTPUT_PATH = "/nobackup_1/users/robben/datasets/dataset.zarr"

START_TIME = pd.Timestamp("2020-01-01 00:00")
END_TIME = pd.Timestamp("2023-12-31 23:59")

source_config = {
    "path": os.path.join(BASE_DATA_DIR, "aws/xy"),
    "interval": 10,  # minutes
    "fill_value": 0.0,
    "chunk_size": 1,
    "shard_size": 1024,
    "vars": {
        # Ground Temp 
        # 'TOB_T_DRYB_MIN5CM_10': {'dtype': np.float16, 'dims': (4,)},
        # 'TOB_Q_T_DRYB_MIN5CM_10': {'dtype': np.float16, 'dims': (4,)},
        # 'TOB_T_DRYB_MIN10CM_10': {'dtype': np.float16, 'dims': (4,)},
        # 'TOB_Q_T_DRYB_MIN10CM_10': {'dtype': np.float16, 'dims': (4,)},
        # 'TOB_T_DRYB_MIN20CM_10': {'dtype': np.float16, 'dims': (4,)},
        # 'TOB_Q_T_DRYB_MIN20CM_10': {'dtype': np.float16, 'dims': (4,)},
        # 'TOB_T_DRYB_MIN50CM_10': {'dtype': np.float16, 'dims': (4,)},
        # 'TOB_Q_T_DRYB_MIN50CM_10': {'dtype': np.float16, 'dims': (4,)},
        # 'TOB_T_DRYB_MIN100CM_10': {'dtype': np.float16, 'dims': (4,)},
        # 'TOB_Q_T_DRYB_MIN100CM_10': {'dtype': np.float16, 'dims': (4,)},

        # Precipitation
        'TOR_RI_PWS_10': {'dtype': np.float16, 'dims': (26,)},
        # 'TOR_Q_RI_PWS_10': {'dtype': np.float16, 'dims': (28,)},
        'TOR_DR_PWS_10': {'dtype': np.float16, 'dims': (26,)},
        # 'TOR_Q_DR_PWS_10': {'dtype': np.float16, 'dims': (34,)},
        'TOR_DR_REGENM_10': {'dtype': np.float16, 'dims': (33,)},
        # 'TOR_Q_DR_REGENM_10': {'dtype': np.float16, 'dims': (33,)},
        'TOR_RI_REGENM_10': {'dtype': np.float16, 'dims': (33,)},       
        # 'TOR_Q_RI_REGENM_10': {'dtype': np.float16, 'dims': (33,)},
        
        # Air pressure
        'TOA_P_NAP_MSL_10': {'dtype': np.float16, 'dims': (40,)},
        # 'TOA_Q_P_NAP_MSL_10': {'dtype': np.float16, 'dims': (40,)},
        'TOA_P_STN_LEVEL_10': {'dtype': np.float16, 'dims': (41,)},
        # 'TOA_Q_P_STN_LEVEL_10': {'dtype': np.float16, 'dims': (41,)},
        'TOA_P_SENSOR_10': {'dtype': np.float16, 'dims': (41,)},
        # 'TOA_Q_P_SENSOR_10': {'dtype': np.float16, 'dims': (41,)},

        # Temp (Air)
        'TOT_T_DRYB_10': {'dtype': np.float16, 'dims': (52,)},
        # 'TOT_Q_T_DRYB_10': {'dtype': np.float16, 'dims': (52,)},
        'TOT_TN_DRYB_10': {'dtype': np.float16, 'dims': (51,)},
        # 'TOT_Q_TN_DRYB_10': {'dtype': np.float16, 'dims': (51,)},
        'TOT_TX_DRYB_10': {'dtype': np.float16, 'dims': (51,)},
        # 'TOT_Q_TX_DRYB_10': {'dtype': np.float16, 'dims': (51,)},
        'TOT_T_DEWP_10': {'dtype': np.float16, 'dims': (52,)},
        # 'TOT_Q_T_DEWP_10': {'dtype': np.float16, 'dims': (52,)},
        'TOT_T_WETB_10': {'dtype': np.float16, 'dims': (49,)},
        # 'TOT_Q_T_WETB_10': {'dtype': np.float16, 'dims': (50,)},
        # 'TOT_T_DEWP_SEA_10': {'dtype': np.float16, 'dims': (3,)},
        # 'TOT_Q_T_DEWP_SEA_10': {'dtype': np.float16, 'dims': (3,)},
        'TOT_TN_10CM_PAST_6H_10': {'dtype': np.float16, 'dims': (37,)},
        # 'TOT_Q_TN_10CM_PAST_6H_10': {'dtype': np.float16, 'dims': (37,)},
        
        # Sea water and waves
        # 'TOD_T_SEAWATER_10': {'dtype': np.float16, 'dims': (3,)},
        # 'TOD_PT_WAVE_10': {'dtype': np.float16, 'dims': (3,)},
        # 'TOD_Q_PT_WAVE_10': {'dtype': np.float16, 'dims': (3,)},
        # 'TOD_A_WAVE_10': {'dtype': np.float16, 'dims': (3,)},
        # 'TOD_Q_A_WAVE_10': {'dtype': np.float16, 'dims': (3,)},

        # Cloud cover
        'TOC_N_LAYER_10': {'dtype': np.float16, 'dims': (31,)},
        # 'TOC_Q_N_LAYER_10': {'dtype': np.float16, 'dims': (34,)},
        'TOC_N1_LAYER_10': {'dtype': np.float16, 'dims': (31,)},
        # 'TOC_Q_N1_LAYER_10': {'dtype': np.float16, 'dims': (34,)},
        'TOC_N2_LAYER_10': {'dtype': np.float16, 'dims': (31,)},
        # 'TOC_Q_N2_LAYER_10': {'dtype': np.float16, 'dims': (34,)},
        'TOC_N3_LAYER_10': {'dtype': np.float16, 'dims': (31,)},  
        # 'TOC_Q_N3_LAYER_10': {'dtype': np.float16, 'dims': (34,)},
        'TOC_H1_LAYER_10': {'dtype': np.float16, 'dims': (31,)},
        # 'TOC_Q_H1_LAYER_10': {'dtype': np.float16, 'dims': (34,)},
        'TOC_H2_LAYER_10': {'dtype': np.float16, 'dims': (31,)},
        # 'TOC_Q_H2_LAYER_10': {'dtype': np.float16, 'dims': (34,)},
        'TOC_H3_LAYER_10': {'dtype': np.float16, 'dims': (31,)},
        # 'TOC_Q_H3_LAYER_10': {'dtype': np.float16, 'dims': (34,)},

        # Wind
        # 'TOW_DDN_10': {'dtype': np.float16, 'dims': (7,)},
        # 'TOW_Q_DDN_10': {'dtype': np.float16, 'dims': (9,)},
        'TOW_DD_10': {'dtype': np.float16, 'dims': (80,)},
        # 'TOW_Q_DD_10': {'dtype': np.float16, 'dims': (81,)},
        'TOW_DD_STD_10': {'dtype': np.float16, 'dims': (80,)},
        # 'TOW_Q_DD_STD_10': {'dtype': np.float16, 'dims': (81,)},
        # 'TOW_DDX_10': {'dtype': np.float16, 'dims': (7,)},
        # 'TOW_Q_DDX_10': {'dtype': np.float16, 'dims': (9,)},
        'TOW_FF_10M_10': {'dtype': np.float16, 'dims': (80,)},
        # 'TOW_Q_FF_10M_10': {'dtype': np.float16, 'dims': (81,)},
        'TOW_FF_10M_STD_10': {'dtype': np.float16, 'dims': (68,)},
        # 'TOW_Q_FF_10M_STD_10': {'dtype': np.float16, 'dims': (70,)},
        'TOW_FF_SENSOR_10': {'dtype': np.float16, 'dims': (80,)},
        # 'TOW_Q_FF_SENSOR_10': {'dtype': np.float16, 'dims': (81,)},
        'TOW_FX_10M_10': {'dtype': np.float16, 'dims': (72,)},
        # 'TOW_Q_FX_10M_10': {'dtype': np.float16, 'dims': (80,)},
        'TOW_FX_10M_MD_10': {'dtype': np.float16, 'dims': (80,)},
        # 'TOW_Q_FX_10M_MD_10': {'dtype': np.float16, 'dims': (81,)},
        'TOW_FX_SENSOR_10': {'dtype': np.float16, 'dims': (80,)},
        # 'TOW_Q_FX_SENSOR_10': {'dtype': np.float16, 'dims': (81,)},
        'TOW_FX_SENSOR_MD_10': {'dtype': np.float16, 'dims': (67,)},
        # 'TOW_Q_FX_SENSOR_MD_10': {'dtype': np.float16, 'dims': (69,)},

        # Zonneschijn & globale straling
        'TOS_SQ_10': {'dtype': np.float16, 'dims': (32,)},
        # 'TOS_Q_SQ_10': {'dtype': np.float16, 'dims': (32,)},
        'TOS_Q_GLOB_10': {'dtype': np.float16, 'dims': (32,)}, 
        # 'TOS_Q_Q_GLOB_10': {'dtype': np.float16, 'dims': (32,)}, 
        'TOS_QX_GLOB_10': {'dtype': np.float16, 'dims': (32,)},
        # 'TOS_Q_QX_GLOB_10': {'dtype': np.float16, 'dims': (32,)},
        'TOS_QN_GLOB_10': {'dtype': np.float16, 'dims': (32,)},
        # 'TOS_Q_QN_GLOB_10': {'dtype': np.float16, 'dims': (32,)}, 
    }
}


def create_time_index_map(start_time: pd.Timestamp, end_time: pd.Timestamp, interval_minutes: int) -> dict:
    # Generate the timestamps at the specified interval
    timestamps = pd.date_range(start=start_time, end=end_time, freq=f'{interval_minutes}min')
    
    # Create a dictionary mapping timestamp to its index
    time_index_map = {timestamp: idx for idx, timestamp in enumerate(timestamps)}

    return time_index_map


def process_variable(var, root, source_config, n_timesteps):
    """Process a single variable and return results without writing to Zarr."""
    print(f'Processing: {var}')
    file_name = f'{var}.parquet'
    path = os.path.join(source_config['path'], file_name)
    var_config = source_config['vars'][var]
    df = pd.read_parquet(path)

    # Get unique locations
    unique_locs = df[['X', 'Y']].drop_duplicates().sort_values(['X', 'Y'])
    n_locations = len(unique_locs)

    if n_locations != var_config['dims'][0]:
        print(f"Warning: for {var}, expected {var_config['dims'][0]} locations, found {n_locations}")

    # Map locations to indices
    loc_to_idx = {tuple(loc): idx for idx, loc in enumerate(unique_locs.itertuples(index=False))}
    
    # Initialize arrays
    data_array = np.full((n_timesteps, n_locations), source_config['fill_value'], dtype=var_config['dtype'])
    time_mask = np.zeros(n_timesteps, dtype=bool)

    # Filter relevant columns and drop duplicates in df
    df = df[['T', 'X', 'Y', 'value']].drop_duplicates(subset=['T', 'X', 'Y'])
    
    # Filter out-of-range timestamps
    valid_times = df['T'] < n_timesteps
    df = df[valid_times]
    
    # Convert locations to indices
    loc_keys = list(zip(df['X'], df['Y']))
    valid_locs = [key in loc_to_idx for key in loc_keys]
    df = df[valid_locs]
    loc_indices = [loc_to_idx[key] for key in loc_keys if key in loc_to_idx]
    
    # Update data array and time mask
    if not df.empty:
        data_array[df['T'], loc_indices] = df['value'].values
        time_mask[df['T']] = True

    # Prepare location array
    loc_array = unique_locs.values.astype(np.uint16)

    # Create main data array
    shape = (n_timesteps, *var_config['dims'])
    chunks = (source_config['chunk_size'], *var_config['dims'])
    shards = (source_config['shard_size'], *var_config['dims'])
    arr = root.create_array(
        var, shape=shape, chunks=chunks, shards=shards, dtype=var_config['dtype']
    )
    arr[:] = data_array
    
    # Create location array
    loc_arr = root.create_array(
        var + '_loc', shape=loc_array.shape, chunks=loc_array.shape, dtype=loc_array.dtype
    )
    loc_arr[:] = loc_array
    
    # Create time mask
    tm_arr = root.create_array(
        var + '_timemask', shape=time_mask.shape, chunks=time_mask.shape, dtype=time_mask.dtype
    )
    tm_arr[:] = time_mask

    print(f'Written {var} to disk')
    
    # return var, data_array, loc_array, time_mask

def main():
    root = zarr.open_group(ZARR_OUTPUT_PATH, mode='a')
    group = root.create_group("aws")

    # Create time index map and get number of timesteps
    time_map = create_time_index_map(START_TIME, END_TIME, source_config['interval'])
    n_timesteps = len(time_map)

    # Add metadata attributes
    group.attrs['start_time'] = str(START_TIME)
    group.attrs['end_time'] = str(END_TIME)
    group.attrs['interval_minutes'] = source_config['interval']
    group.attrs['fill_value'] = source_config['fill_value']
    group.attrs['creation_date'] = str(pd.Timestamp.now())

    # Process variables in parallel
    num_workers = min(len(source_config['vars']), mp.cpu_count())
    print(f"Using {num_workers} workers for processing")
    
    with mp.Pool(processes=num_workers) as pool:
        # Prepare arguments for parallel processing
        process_func = partial(
            process_variable,
            root=group,
            n_timesteps=n_timesteps,
            source_config=source_config
        )
        pool.map(process_func, source_config['vars'].keys())

    # # Write results to Zarr
    # for var, data_array, loc_array, time_mask in results:
    #     var_config = source_config['vars'][var]
        
    print(root.tree())

if __name__ == '__main__':
    main()