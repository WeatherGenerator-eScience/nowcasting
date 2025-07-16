import zarr
import os
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
ZARR_OUTPUT_PATH = "/nobackup_1/users/robben/datasets/dataset_regrid.zarr"

START_TIME = pd.Timestamp("2020-01-01 00:00")
END_TIME = pd.Timestamp("2023-12-31 23:59")

# --- Data Source Configuration ---
source_config = {
    "path": os.path.join(BASE_DATA_DIR, "satellite/L_2"),
    "interval": 15,  # minutes
    "fill_value": 0.0,
    "chunk_size": 1,
    "shard_size": 512,
    "vars": {
        'cldmask': {'dtype': np.bool_, 'dims': (384, 352)}, # (126, 250)},
        'cot': {'dtype': np.float16, 'dims': (384, 352)}, # (126, 250)},
        'cot_unc': {'dtype': np.float16, 'dims': (384, 352)}, # (126, 250)},
        'cph': {'dtype': np.float16, 'dims': (384, 352)}, # (126, 250)},
        'cre': {'dtype': np.float16, 'dims': (384, 352)}, # (126, 250)},
        'cre_unc': {'dtype': np.float16, 'dims': (384, 352)}, # (126, 250)},
        'cth': {'dtype': np.float16, 'dims': (384, 352)}, # (126, 250)},
        'ctp': {'dtype': np.float16, 'dims': (384, 352)}, # (126, 250)},
        'ctt': {'dtype': np.float16, 'dims': (384, 352)}, # (126, 250)},
        'ctype': {'dtype': np.uint8, 'dims': (384, 352)}, # (126, 250)},
        'ctype_nwc': {'dtype': np.uint8, 'dims': (384, 352)}, # (126, 250)},
        'cwp': {'dtype': np.float16, 'dims': (384, 352)}, # (126, 250)},
        'cwp_unc': {'dtype': np.float16, 'dims': (384, 352)}, # (126, 250)},
        'irrad_toa': {'dtype': np.float16, 'dims': (384, 352)}, # (126, 250)},
        'precip': {'dtype': np.float16, 'dims': (384, 352)}, # (126, 250)},
        'precip_ir': {'dtype': np.float16, 'dims': (384, 352)}, # (126, 250)},
        'sds': {'dtype': np.float16, 'dims': (384, 352)}, # (126, 250)},
        'sds_b1': {'dtype': np.float16, 'dims': (384, 352)}, # (126, 250)},
        'sds_b2': {'dtype': np.float16, 'dims': (384, 352)}, # (126, 250)},
        'sds_b3': {'dtype': np.float16, 'dims': (384, 352)}, # (126, 250)},
        'sds_b4': {'dtype': np.float16, 'dims': (384, 352)}, # (126, 250)},
        'sds_b5': {'dtype': np.float16, 'dims': (384, 352)}, # (126, 250)},
        'sds_cs': {'dtype': np.float16, 'dims': (384, 352)}, # (126, 250)},
        'sds_cs_b1': {'dtype': np.float16, 'dims': (384, 352)}, # (126, 250)},
        'sds_cs_b2': {'dtype': np.float16, 'dims': (384, 352)}, # (126, 250)},
        'sds_cs_b3': {'dtype': np.float16, 'dims': (384, 352)}, # (126, 250)},
        'sds_cs_b4': {'dtype': np.float16, 'dims': (384, 352)}, # (126, 250)},
        'sds_cs_b5': {'dtype': np.float16, 'dims': (384, 352)}, # (126, 250)},
        'sds_dir': {'dtype': np.float16, 'dims': (384, 352)}, # (126, 250)},
        'sds_dir_cs': {'dtype': np.float16, 'dims': (384, 352)}, # (126, 250)},
        'sunazi': {'dtype': np.float16, 'dims': (384, 352)}, # (126, 250)},
        'sunz': {'dtype': np.float16, 'dims': (384, 352)}, # (126, 250)},
        'trs': {'dtype': np.float16, 'dims': (384, 352)}, # (126, 250)},
        'trs_cs': {'dtype': np.float16, 'dims': (384, 352)}, # (126, 250)},
    }
}

BBOX = (48.86491832493661, 55.97360229491834, 0.0, 10.91708999269145)

def get_num_timesteps(start_time, end_time, interval):
    """Calculate the number of timestamps between start_time and end_time."""
    delta = end_time - start_time
    return int(delta.total_seconds() // (interval * 60)) + 1

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
                yield fileobj, ts

def crop_to_bbox(data_2D, lat_2D, lon_2D, bbox):
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
        data_2D[row_min:row_max+1, col_min:col_max+1],
        lat_2D[row_min:row_max+1, col_min:col_max+1],
        lon_2D[row_min:row_max+1, col_min:col_max+1],
        (slice(row_min, row_max+1), slice(col_min, col_max+1))
    )

def process_nc_file(file_obj, source_config, slices=None, transform=None):
    vars_to_process = source_config['vars']
    results = {}

    with Dataset('name', "r", memory=file_obj.getvalue()) as nc_file:
        for var_name, config in vars_to_process.items():
            if var_name in nc_file.variables:
                try:
                    # Load the variable data
                    var = nc_file.variables[var_name]
                    data = var[:]
                    
                    # Check for scale_factor attribute and apply it if present
                    scale_factor = getattr(var, 'scale_factor', 1.0)
                    data = data * scale_factor
                    
                    # Reshape data if it has shape (1, height, width)
                    if data.shape[0] == 1 and len(data.shape) == 3:
                        data = data[0, :, :]
                    if slices:
                        data = data[slices[0], slices[1]]
                    if transform:
                        data = transform(data)

                    # Cast the data to the specified dtype
                    data = data.astype(config['dtype'])
                    
                    # Save the data to the results dictionary
                    results[var_name] = data
                except Exception as e:
                    print(f"Error processing variable {var_name}: {e}")
            else:
                print(f"Warning: Variable {var_name} not found in the NetCDF file.")
        
        return results

def extract_and_store_latlon(source_config, root, bbox=BBOX):
    """Extracts lat/lon from first available file, crops to bbox, and stores in Zarr."""
    source_files = filter_tar_files_by_time(source_config['path'], START_TIME, END_TIME)
    
    for file_path in source_files[:1]:  # Only process first file
        for fileobj, timestamp in generate_members_tar(file_path):
            try:
                with Dataset('name', "r", memory=fileobj.getvalue()) as nc_file:
                    if 'lat' not in nc_file.variables or 'lon' not in nc_file.variables:
                        print("Warning: lat/lon variables not found in file")
                        continue
                        
                    lat = nc_file.variables['lat'][:]
                    lon = nc_file.variables['lon'][:]
                    dummy_data = np.zeros_like(lat)
                    
                    # Crop to bounding box
                    _, cropped_lat, cropped_lon, slices = crop_to_bbox(
                        dummy_data, lat, lon, bbox
                    )
                    
                    # Store in Zarr
                    root['lat'][...] = cropped_lat.astype(np.float16)
                    root['lon'][...] = cropped_lon.astype(np.float16)
                    print("Successfully stored cropped lat/lon arrays")
                    return slices
                    
            except Exception as e:
                print(f"Error processing lat/lon: {e}")
                continue
    
    raise RuntimeError("Failed to extract lat/lon from any file")

def process_source_sequentially(root, source_config, time_map):
    shard_size = source_config['shard_size']
    source_files = filter_tar_files_by_time(source_config['path'], START_TIME, END_TIME)

    buffers = {var: {} for var in source_config['vars'].keys()}
    for var in buffers:
        buffers[var] = {
            "shard_map": {},  # shard_index: (indices, data)
        }

    # Get crop slices from lat/lon
    try:
        slices = extract_and_store_latlon(source_config, root)
    except Exception as e:
        print(f"Critical error in lat/lon processing: {e}")
        return

    downscale_factor = 2
    transform = regrid_transform(root['lat'][:], root['lon'][:], downscale_factor=downscale_factor)
    root.attrs['downscale_factor'] = downscale_factor

    preivous_idx = 0
    for file_path in source_files:
        print(file_path)

        for fileobj, timestamp in generate_members_tar(file_path):

            index = time_map.get(timestamp)
            if index is None: continue

            result = process_nc_file(fileobj, source_config, slices=slices, transform=transform)
            if result is None: continue
            
            shard_idx = index // shard_size
            shard_idx_previous = preivous_idx // shard_size

            for var, data in result.items():
                if var not in source_config['vars']: continue

                # Initialize shard buffer if needed
                if shard_idx not in buffers[var]["shard_map"]:
                    buffers[var]["shard_map"][shard_idx] = {"indices": [], "data": []}

                # Add to shard buffer
                buffers[var]["shard_map"][shard_idx]["indices"].append(index)
                buffers[var]["shard_map"][shard_idx]["data"].append(data)

                # Write the previous shard when moved on to the next
                if shard_idx_previous < shard_idx:
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
    group = root.create_group("sat_l2")

    # Create datasets for variables
    time_map = create_time_index_map(START_TIME, END_TIME, source_config['interval'])

    # Add metadata attributes
    group.attrs['start_time'] = str(START_TIME)
    group.attrs['end_time'] = str(END_TIME)
    group.attrs['interval_minutes'] = source_config['interval']
    group.attrs['fill_value'] = source_config['fill_value']
    group.attrs['bbox'] = BBOX
    group.attrs['creation_date'] = str(pd.Timestamp.now())

    for var, var_config in source_config['vars'].items():
        shape = (len(time_map), *var_config['dims'])
        chunks = (source_config['chunk_size'], *var_config['dims'])
        
        # Calculate shard shape as multiple of chunk shape
        chunk_size = source_config['chunk_size']
        shard_size = source_config['shard_size']
        if shard_size % chunk_size != 0:
            # Round up to nearest multiple
            shard_size = ((shard_size // chunk_size) + 1) * chunk_size
            
        shards = (shard_size, *var_config['dims'])

        group.create_array(
            name=var, shape=shape, chunks=chunks, shards=shards, dtype=var_config['dtype']
        )

    # Create datasets for lat/lon
    group.create_array('lat', shape=(126, 250), dtype=np.float16)
    group.create_array('lon', shape=(126, 250), dtype=np.float16)

    group.create_array(
        name='time_mask', 
        shape=(len(time_map),), 
        chunks=(len(time_map),),
        dtype=bool, 
    )
    group['time_mask'][...] = False  # Initialize to all missing
    
    process_source_sequentially(group, source_config, time_map)
    print(root.tree())

if __name__ == '__main__':
    main()