import zarr
import os
import warnings
import numpy as np
import pandas as pd
import re
import glob
import tarfile
import io
from datetime import datetime
from netCDF4 import Dataset

# --- Configuration ---
BASE_DATA_DIR = "/nobackup_1/users/robben/full_dataset"
ZARR_OUTPUT_PATH = "/nobackup_1/users/robben/datasets/dataset_test2.zarr"

START_TIME = pd.Timestamp("2020-10-01 00:00")
END_TIME = pd.Timestamp("2020-10-31 23:59")

# --- Data Source Configuration ---
source_config = {
    "path": os.path.join(BASE_DATA_DIR, "harmonie"),
    "interval": 6 * 60,  # minutes
    "fill_value": 0.0,
    "chunk_size": 1,
    "shard_size": 32,
    "forecast_range": (3, 13),
    "vars": {
        'SHTFL_GDS0_HTGL': {'dtype': np.float16, 'dims': (384, 352)},
        'PRES_GDS0_GPML': {'dtype': np.float32, 'dims': (384, 352)},
        'PRES_GDS0_HTGL': {'dtype': np.float32, 'dims': (384, 352)},
        'GP_GDS0_HTGL': {'dtype': np.float16, 'dims': (384, 352)},
        'TMP_GDS0_HTGL': {'dtype': np.float16, 'dims': (6, 384, 352)},
        'DPT_GDS0_HTGL': {'dtype': np.float16, 'dims': (384, 352)},
        'VIS_GDS0_HTGL': {'dtype': np.float16, 'dims': (384, 352)},
        'U_GRD_GDS0_HTGL': {'dtype': np.float16, 'dims': (5, 384, 352)},
        'V_GRD_GDS0_HTGL': {'dtype': np.float16, 'dims': (5, 384, 352)},
        'R_H_GDS0_HTGL': {'dtype': np.float16, 'dims': (384, 352)},
        'A_PCP_GDS0_HTGL_acc': {'dtype': np.float16, 'dims': (384, 352), 'deacc': True},
        'WEASD_GDS0_HTGL': {'dtype': np.float16, 'dims': (384, 352)},
        'MIXHT_GDS0_HTGL': {'dtype': np.float16, 'dims': (384, 352)},
        'T_CDC_GDS0_HTGL': {'dtype': np.float16, 'dims': (384, 352)},
        'L_CDC_GDS0_HTGL': {'dtype': np.float16, 'dims': (384, 352)},
        'M_CDC_GDS0_HTGL': {'dtype': np.float16, 'dims': (384, 352)},
        'H_CDC_GDS0_HTGL': {'dtype': np.float16, 'dims': (384, 352)},
        'LAND_GDS0_HTGL': {'dtype': np.bool_, 'dims': (384, 352)},
        'NSWRS_GDS0_HTGL_acc': {'dtype': np.float32, 'dims': (384, 352), 'deacc': True},
        'NLWRS_GDS0_HTGL_acc': {'dtype': np.float32, 'dims': (384, 352), 'deacc': True},
        'G_RAD_GDS0_HTGL_acc': {'dtype': np.float32, 'dims': (384, 352), 'deacc': True},
        'SHTFL_GDS0_HTGL_acc': {'dtype': np.float32, 'dims': (384, 352), 'deacc': True},
        'KNMI_var_132_acc': {'dtype': np.float32, 'dims': (384, 352), 'deacc': True},
        'KNMI_var_162_acc': {'dtype': np.float16, 'dims': (384, 352)},
        'KNMI_var_163_acc': {'dtype': np.float16, 'dims': (384, 352)},
        'KNMI_var_181': {'dtype': np.float16, 'dims': (384, 352)},
        'KNMI_var_181_acc': {'dtype': np.float16, 'dims': (384, 352), 'deacc': True},
        'KNMI_var_184': {'dtype': np.float16, 'dims': (384, 352)},
        'KNMI_var_184_acc': {'dtype': np.float16, 'dims': (384, 352), 'deacc': True},
        'KNMI_var_186_entatm': {'dtype': np.float32, 'dims': (384, 352)},
        'KNMI_var_201': {'dtype': np.float16, 'dims': (384, 352)},
        'KNMI_var_201_entatm': {'dtype': np.float16, 'dims': (384, 352)},
        'KNMI_var_201_acc': {'dtype': np.float16, 'dims': (384, 352), 'deacc': True},
    }
}

BBOX = (48.86491832493661, 55.97360229491834, 0.0, 10.91708999269145)


def create_time_index_map(start_time: pd.Timestamp, end_time: pd.Timestamp, interval_minutes: int) -> dict:
    # Generate the timestamps at the specified interval
    timestamps = pd.date_range(start=start_time, end=end_time, freq=f'{interval_minutes}min')
    
    # Create a dictionary mapping timestamp to its index
    time_index_map = {timestamp: idx for idx, timestamp in enumerate(timestamps)}

    return time_index_map

def filter_tar_files_by_time(source_dir, start_time, end_time):
    """
    Returns a list of tar files in `source_dir` that have a timestamp between start_time and end_time.
    Sorted chronologically by detected timestamp.
    """
    start_time = start_time.replace(hour=0, minute=0, second=0, microsecond=0)
    end_time = end_time.replace(hour=23, minute=59, second=59, microsecond=999999)

    all_files = glob.glob(os.path.join(source_dir, '*.tar'))
    all_files.sort()
    filtered_files = []

    for file in all_files:
        timestamp = get_timestamp_from_filename(file)
        if timestamp and start_time <= timestamp <= end_time:
            filtered_files.append((timestamp, file))

    # Sort by timestamp
    filtered_files.sort(key=lambda x: x[0])
    return [file for _, file in filtered_files]

def get_timestamp_from_filename(filename):
    """
    Extract the most relevant timestamp from a filename using multiple regex patterns.
    Tries in order: YYYYMMDDHHMMSS, YYYYMMDDHHMM, YYYYMMDD.
    Returns: pd.Timestamp or None
    """
    basename = os.path.basename(filename)

    patterns = [
        (r"_(\d{8}T\d{6})", "%Y%m%dT%H%M%S"), # Match e.g. _20200101T001500
        (r"_(\d{14})",       "%Y%m%d%H%M%S"), # Match e.g. _20200101001500
        (r"_(\d{12})",       "%Y%m%d%H%M"),   # Match e.g. _202001010015
        (r"_(\d{8})",        "%Y%m%d"),       # Match e.g. _20200101
    ]

    for regex, date_fmt in patterns:
        match = re.search(regex, basename)
        if match:
            try:
                dt = datetime.strptime(match.group(1), date_fmt)
                return pd.Timestamp(dt)
            except ValueError:
                continue
    return None

def get_forecast_step(filename):
    match = re.search(r'_(\d+)\.nc$', filename)
    if match:
        return int(match.group(1)) // 100
    
def regrid_transform(src_lat, src_lon, downscale_factor=2):

    import pyproj
    import wradlib as wrl

    # Define target grid parameters
    target_corners = {
        'll': (0, 49.362064),    # (lon, lat)
        'ul': (0, 55.973602),
        'ur': (10.856453, 55.388973),
        'lr': (9.0093, 48.8953)
    }
    org_height, org_width = 765, 700
    pad_height, pad_width = 768, 704
    
    # Create projection system
    proj_str = (
        "+proj=stere +lat_0=90 +lon_0=0 +lat_ts=60 "
        "+a=6378.14 +b=6356.75 +x_0=0 +y_0=0"
    )
    proj = pyproj.Proj(proj_str)

    # Project target corners to XY coordinates
    x_ul, y_ul = proj(*target_corners['ul'])
    x_lr, y_lr = proj(*target_corners['lr'])
    
    # Calculate grid steps
    dx = (x_lr - x_ul) / (org_width - 1)
    dy = (y_lr - y_ul) / (org_height - 1)

    # Create padded grid in projected space
    x_grid = np.linspace(x_ul, x_ul + dx * (pad_width - 1), pad_width)
    y_grid = np.linspace(y_ul, y_ul + dy * (pad_height - 1), pad_height)
    x_2d_grid, y_2d_grid = np.meshgrid(x_grid, y_grid)
    s_grid_2d = np.shape(x_2d_grid)

    grid_points_xy = np.squeeze(
        np.array(list(zip([np.ravel(x_2d_grid)], [np.ravel(y_2d_grid)])))
    )

    s_zip = np.shape(grid_points_xy)
    lons = []
    lats = []
    for i in range(0, s_zip[1]):
        lon, lat = proj(grid_points_xy[0, i], grid_points_xy[1, i], inverse=True)
        lons.append(lon)
        lats.append(lat)

    lons = np.asarray(lons)
    lats = np.asarray(lats)
    trg_lon = np.reshape(lons, (s_grid_2d[0], s_grid_2d[1]))
    trg_lat = np.reshape(lats, (s_grid_2d[0], s_grid_2d[1]))

    # Apply downscaling
    if downscale_factor > 1:
        trg_lon = trg_lon[::downscale_factor, ::downscale_factor]
        trg_lat = trg_lat[::downscale_factor, ::downscale_factor]

    print('Target latlon shape:', trg_lat.shape, trg_lon.shape)
    print('Source latlon shape:', src_lat.shape, src_lon.shape)
    
    src_coords = np.vstack((src_lat.ravel(), src_lon.ravel())).T
    trg_coords = np.vstack((trg_lat.ravel(), trg_lon.ravel())).T

    interpolator = wrl.ipol.Idw(
        src_coords, trg_coords, nnearest=1
    ) 
    
    def transform(data):
        # Flatten target grid for interpolation
        data_flat = data.ravel()
        
        # Interpolate using nearest neighbor
        interp_data = interpolator(data_flat)
        
        # Reshape to target grid dimensions
        return interp_data.reshape(trg_lat.shape)
    
    return transform

def generate_members_tar(file_path_or_obj, is_fileobj=False):
    with tarfile.open(name=file_path_or_obj if not is_fileobj else None,
                        fileobj=file_path_or_obj if is_fileobj else None,  
                        mode='r:*') as tar:
        for member in sorted(tar.getmembers(), key=lambda member_info: member_info.name):
            tar_extracted = tar.extractfile(member)
            fileobj = io.BytesIO(tar_extracted.read())
            tar_extracted.close()

            if member.name.endswith('.tar'):
                yield from generate_members_tar(fileobj, is_fileobj=True)
            else:
                ts = get_timestamp_from_filename(member.name)
                fs = get_forecast_step(member.name)

                yield fileobj, ts, fs

def crop_to_bbox(lat_2D, lon_2D, bbox):
    min_lat, max_lat, min_lon, max_lon = bbox
    mask = (
        (lat_2D >= min_lat) & (lat_2D <= max_lat) &
        (lon_2D >= min_lon) & (lon_2D <= max_lon)
    )
    rows, cols = np.where(mask)
    if len(rows) == 0 or len(cols) == 0:
        raise ValueError("No overlap with bounding box!")
    row_min, row_max = rows.min(), rows.max()
    col_min, col_max = cols.min(), cols.max()
    return (
        lat_2D[row_min:row_max+1, col_min:col_max+1],
        lon_2D[row_min:row_max+1, col_min:col_max+1],
        (slice(row_min, row_max+1), slice(col_min, col_max+1))
    )

def process_nc_file(file_obj, source_config, slices=None, forecast_step=None, transform=None):
    results = {}

    with Dataset('name', "r", memory=file_obj.getvalue()) as nc_file:
        for var_name, config in source_config['vars'].items():
            if source_config['vars'][var_name].get('deacc', False):
                var_name_ = var_name + f"{forecast_step}h"
            else:
                var_name_ = var_name

            if var_name_ in nc_file.variables:
                try:
                    # Load the variable data
                    var = nc_file.variables[var_name_]
                    data = var[:]
                    
                    # Reshape data if it has shape (1, height, width)
                    if data.shape[0] == 1 and len(data.shape) == 3:
                        data = data[0, :, :]

                    # print('Original data shape:', data.shape)
                    if slices and len(data.shape) == 3:
                        data = data[:, slices[0], slices[1]]
                        # print('Sliced data shape:', data.shape)
                        if transform:
                            data = np.array([transform(d) for d in data])
                            # print('Transformed data shape:', data.shape)

                    elif slices and len(data.shape) == 2:
                        data = data[slices[0], slices[1]]
                        # print('Sliced data shape:', data.shape)
                        if transform:
                            data = transform(data)
                            # print('Transformed data shape:', data.shape)
                    
                    # Cast the data to the specified dtype
                    data = data.astype(config['dtype'])

                    # Catch overflow warning as an exception
                    with warnings.catch_warnings(record=True) as w:
                        warnings.simplefilter("always", RuntimeWarning)
                        data = data.astype(config['dtype'])

                        for warning in w:
                            if issubclass(warning.category, RuntimeWarning):
                                print(f"Warning while casting variable '{var_name}': {warning.message}")
                    
                    # Save the data to the results dictionary
                    results[var_name] = data
                except Exception as e:
                    print(f"Error processing variable {var_name}: {e}")
                    raise
            else:
                print(f"Warning: Variable {var_name} not found in the NetCDF file.")
        
        return results

def extract_and_store_latlon(source_files, root, bbox=BBOX):
    """Extracts lat/lon from first available file, crops to bbox, and stores in Zarr."""
    for file_path in source_files:  # Only process first file
        for fileobj, _, _ in generate_members_tar(file_path):
            try:
                with Dataset('name', "r", memory=fileobj.getvalue()) as nc_file:
                    if 'g0_lat_0' not in nc_file.variables or 'g0_lon_1' not in nc_file.variables:
                        print("Warning: lat/lon variables not found in file")
                        continue
                        
                    lat = nc_file.variables['g0_lat_0'][:]
                    lon = nc_file.variables['g0_lon_1'][:]

                    if len(lat.shape) < 2 or len(lon.shape) < 2:
                        lon, lat = np.meshgrid(lon, lat)

                    print(lon.shape, lat.shape)
                    
                    # Crop to bounding box
                    cropped_lat, cropped_lon, slices = crop_to_bbox(
                        lat, lon, bbox
                    )

                    # Store in Zarr
                    root['lat'][...] = cropped_lat.astype(np.float16)
                    root['lon'][...] = cropped_lon.astype(np.float16)
                    print("Successfully stored cropped lat/lon arrays")
                    return slices
                        
            except Exception as e:
                print(f"Error processing lat/lon: {e}")
                return
    
    raise #RuntimeError("Failed to extract lat/lon from any file")

def process_source_sequentially(root, source_config, time_map):
    source_files = filter_tar_files_by_time(source_config['path'], START_TIME, END_TIME)
    shard_size = source_config['shard_size']
    f_start, f_end = source_config['forecast_range']
    f_len = f_end - f_start

    buffers = {var: {} for var in source_config['vars'].keys()}
    for var in buffers:
        buffers[var] = {
            "shard_map": {},  # shard_index: (indices, data)
        }

    deacc_vars = {v for v, cfg in source_config['vars'].items() if cfg.get('deacc', False)}
    print(deacc_vars)

    # Get crop slices from lat/lon
    try:
        slices = extract_and_store_latlon(source_files, root)
    except Exception as e:
        print(f"Critical error in lat/lon processing: {e}")
        raise

    downscale_factor = 2
    transform = regrid_transform(root['lat'][:], root['lon'][:], downscale_factor=downscale_factor)
    root.attrs['downscale_factor'] = downscale_factor

    temp_arrays = {}
    for var, cfg in source_config['vars'].items():
        dims = cfg['dims']
        dtype = cfg['dtype']
        fill = source_config['fill_value']
        length = f_len + 1 if var in deacc_vars else f_len
        temp_arrays[var] = np.full((length, *dims), fill, dtype=dtype)

    preivous_idx = 0
    for file_path in source_files:
        print(file_path)
        for fileobj, timestamp, forecast_step in generate_members_tar(file_path):
            if forecast_step < f_start or forecast_step > f_end:
                continue

            index = time_map.get(timestamp)
            if (index - preivous_idx) > 1:
                preivous_idx = index
            if index is None: continue

            result = process_nc_file(fileobj, source_config, slices=slices, forecast_step=forecast_step, transform=transform)
            if result is None: continue

            forecast_step -= f_start

            # Add data to temp array
            for var, data in result.items():
                if var not in temp_arrays:
                    continue
                    
                if forecast_step == f_len:
                    if var in deacc_vars:
                        temp_arrays[var][forecast_step] = data
                else:
                    temp_arrays[var][forecast_step] = data
                    
            # When full forecast is obtained
            if forecast_step == f_len:
                shard_idx = index // shard_size
                shard_idx_previous = preivous_idx // shard_size

                out_batch = {}
                for var in temp_arrays:
                    if var in deacc_vars:
                        arr = temp_arrays[var]               # (window_len+1,…)
                        out_batch[var] = arr[1:] - arr[:-1]  # (window_len,…)
                    else:
                        out_batch[var] = temp_arrays[var]

                for var, data in out_batch.items():
                    if shard_idx not in buffers[var]["shard_map"]:
                        buffers[var]["shard_map"][shard_idx] = {"indices": [], "data": []}

                    # Add to shard buffer
                    buffers[var]["shard_map"][shard_idx]["indices"].append(index)
                    buffers[var]["shard_map"][shard_idx]["data"].append(data)
                    temp_arrays[var] = np.full(temp_arrays[var].shape, source_config['fill_value'], dtype=data.dtype)

                    if shard_idx_previous < shard_idx and len(buffers[var]["shard_map"][shard_idx_previous]["data"])>0:
                        write_shard(root[var], buffers[var]["shard_map"][shard_idx_previous])
                        # buffers[var]["shard_map"][shard_idx_previous]["indices"].clear()
                        buffers[var]["shard_map"][shard_idx_previous]["data"].clear()
                
                preivous_idx = index
            
    # Process remaining partial shards
    for var in buffers:
        # Write any complete shards in shard_map
        for shard_idx, buffer in buffers[var]["shard_map"].items():
            if len(buffer["data"]) > 0:
                write_shard(root[var], buffer)

        
    # Update time mask
    processed_indices = set()
    for var in buffers:
        for buffer in buffers[var]["shard_map"].values():
            processed_indices.update(buffer["indices"])
    
    if processed_indices:
        root['time_mask'][np.array(list(processed_indices))] = True

def write_shard(dataset, buffer):
    """Write a full shard or partial shard to Zarr"""
    if not buffer["indices"]:
        return
        
    indices = np.array(buffer["indices"])
    batch_data = np.stack(buffer["data"])
    
    # Sort by index for contiguous writing
    sort_order = np.argsort(indices)
    sorted_indices = indices[sort_order]
    sorted_data = batch_data[sort_order]
    
    # Write in contiguous blocks where possible
    start_idx = 0
    while start_idx < len(sorted_indices):
        end_idx = start_idx + 1
        # Find contiguous block
        while (end_idx < len(sorted_indices)) and \
              (sorted_indices[end_idx] == sorted_indices[end_idx-1] + 1):
            end_idx += 1
            
        # Write contiguous block
        block_start = sorted_indices[start_idx]
        block_end = sorted_indices[end_idx-1] + 1
        dataset[block_start:block_end] = sorted_data[start_idx:end_idx]
        
        start_idx = end_idx

def main():
    root = zarr.open_group(ZARR_OUTPUT_PATH, mode='a')
    group = root.create_group('harmonie', overwrite=True)
    print(f"Zarr store initialized at {ZARR_OUTPUT_PATH}")

    # Create datasets for variables
    time_map = create_time_index_map(START_TIME, END_TIME, source_config['interval'])

    # Add metadata attributes
    group.attrs['start_time'] = str(START_TIME)
    group.attrs['end_time'] = str(END_TIME)
    group.attrs['interval_minutes'] = source_config['interval']
    group.attrs['forecast_range'] = source_config['forecast_range']
    group.attrs['fill_value'] = source_config['fill_value']
    group.attrs['bbox'] = BBOX
    group.attrs['creation_date'] = str(pd.Timestamp.now())

    for var, var_config in source_config['vars'].items():
        forecast_len = source_config['forecast_range'][1] - source_config['forecast_range'][0]
        shape = (len(time_map), forecast_len, *var_config['dims'])
        chunks = (source_config['chunk_size'], forecast_len, *var_config['dims'])
        
        # Calculate shard shape as multiple of chunk shape
        chunk_size = source_config['chunk_size']
        shard_size = source_config['shard_size']
        if shard_size % chunk_size != 0:
            # Round up to nearest multiple
            shard_size = ((shard_size // chunk_size) + 1) * chunk_size
            
        shards = (shard_size, forecast_len, *var_config['dims'])

        group.create_array(
            name=var, shape=shape, chunks=chunks, shards=shards, dtype=var_config['dtype']
        )

    # Create datasets for lat/lon
    group.create_array('lat', shape=(388, 377), dtype=np.float16)
    group.create_array('lon', shape=(388, 377), dtype=np.float16)

    group.create_array(
        name='time_mask', 
        shape=(len(time_map),), 
        chunks=(len(time_map),),
        dtype=bool, 
    )
    group['time_mask'][...] = False  # Initialize to all missing
    
    process_source_sequentially(group, source_config, time_map)
    # print(root.tree())

if __name__ == '__main__':
    main()