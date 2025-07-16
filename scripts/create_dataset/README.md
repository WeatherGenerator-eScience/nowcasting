# Data Processing Pipeline for Nowcasting Dataset

This folder contains Python scripts designed to process and transform various environmental datasets, including Automatic Weather Station (AWS) data, HARMONIE model outputs, Radar data, and Satellite Level 1.5 and Level 2 products, into an optimized Zarr format. These scripts serve primarily to transparently document the data processing steps undertaken to create the final datasets, rather than prescribing a specific input data repository structure.

## 1\. Overview

This collection of Python scripts is a pipeline for processing various meteorological and environmental datasets into a unified, high-performance Zarr archive. Each script is tailored to a specific data type and applies necessary cleaning, transformation, and spatial/temporal alignment steps.

The primary goal of providing these scripts is to offer transparency into the data preparation process for analysis, particularly for machine learning applications. While the raw input data itself is not publicly available, these scripts illustrate the methodology and transformations applied.

The pipeline components include:

  * **`aws.py`**: Processes pre-gridded Automatic Weather Station (AWS) data.
  * **`harmonie.py`**: Handles HARMONIE numerical weather prediction model output.
  * **`radar.py`**: Processes various radar products (RTCOR, RFCOR, ETH), including specialized transformations.
  * **`sat_l1p5.py`**: Processes Level 1.5 satellite imagery (Infrared and Visible bands), involving spatial regridding.
  * **`sat_l2.py`**: Processes Level 2 satellite products (e.g., cloud properties, precipitation estimates), also performing regridding.

All scripts contribute data to a single Zarr archive, designed for efficient, array-based access and analysis, commonly indexed by a common time dimension.

## 2\. Prerequisites

  * **Python 3.11+**
  * **Required Python Libraries:** These are essential for running the scripts. It is recommended to install them within a virtual environment.
      * `pandas`
      * `numpy`
      * `zarr`
      * `netCDF4` (for HARMONIE and Satellite data processing)
      * `h5py` (for Radar data processing)
      * `wradlib` (implied for radar/satellite regridding and processing, e.g., in `sat_l1p5.py`)
      * `pyproj` (for coordinate transformations, used in regridding)
      * `numba` (for performance optimizations, e.g., in `radar.py`)

## 3\. Script Purpose & Structure

These scripts are provided as a single folder. They are meant to be run individually to demonstrate the processing logic for different data sources. The scripts are self-contained in terms of their processing logic for their respective data types.

The input data paths (`BASE_DATA_DIR`) are configured *within* each script. Users of these scripts are not expected to place their raw data in any specific relative location to the scripts themselves. Instead, they should adjust the `BASE_DATA_DIR` and other path variables within the scripts to point to their own data storage.

A conceptual structure for the input data, as anticipated by the scripts' default configurations, might look like this (though users are free to organize their actual data as they see fit and adjust paths accordingly):

```
/path/to/your/full_dataset/     # This would be your BASE_DATA_DIR
├── aws/
│   └── latlon/                 # Contains AWS with the original Lat Lon coordinates (e.g., Parquet files)
│   └── xy/                     # Contains AWS data already transformed to X,Y coordinates (e.g., Parquet files)
├── harmonie/                   # Contains HARMONIE model output (e.g., .tar archives of NetCDF files)
├── radar/
│   ├── RTCOR/                  # Contains RTCOR radar product (e.g., .tar archives of HDF5 files)
│   ├── RFCOR/                  # Contains RFCOR radar product
│   └── ETH/                    # Contains ETH radar product
└── satellite/
    ├── L_1p5/                  # Contains Level 1.5 satellite data (e.g., .tar archives of NetCDF files)
    └── L_2/                    # Contains Level 2 satellite data
```

The primary output (`ZARR_OUTPUT_PATH`) is a Zarr dataset, which encapsulates all processed variables.

## 4\. Processing Steps (Script Usage)

Each script is executed individually. Before running, **you must open each script and configure the paths and time ranges** to match your data setup and desired processing period.

### AWS Data Processing (`aws.py`)

This script processes AWS data that has already been transformed to a gridded (X, Y) format. It reads Parquet files and stores the data as arrays within the main Zarr dataset.

1.  **Configure `aws.py`**:
      * Open `scripts/aws.py`.
      * Adjust `BASE_DATA_DIR` to point to the root of your raw data.
      * Adjust `ZARR_OUTPUT_PATH` to specify where the final Zarr dataset will be created/updated.
      * Set `START_TIME` and `END_TIME` for the desired processing window.
      * Review `source_config` for input path, interval (minutes), fill value, and variable definitions.
2.  **Run the script**:
    ```bash
    python scripts/aws.py
    ```

### HARMONIE Data Processing (`harmonie.py`)

This script processes HARMONIE model output, typically found in NetCDF files archived within `.tar` files. It extracts specified variables and adds them to the Zarr dataset, along with latitude and longitude arrays.

1.  **Configure `harmonie.py`**:
      * Open `scripts/harmonie.py`.
      * Adjust `BASE_DATA_DIR`, `ZARR_OUTPUT_PATH`, `START_TIME`, and `END_TIME`.
      * Review `source_config` for input path, interval, fill value, forecast range, and variable definitions. Pay attention to `BBOX` for spatial context.
2.  **Run the script**:
    ```bash
    python scripts/harmonie.py
    ```

### Radar Data Processing (`radar.py`)

This script processes various radar products (RTCOR, RFCOR, ETH), typically HDF5 files within `.tar` archives. It applies product-specific transformations (e.g., conversion to rain rate, antialiasing, clutter score computation) before storing them in Zarr.

1.  **Configure `radar.py`**:
      * Open `scripts/radar.py`.
      * Adjust `BASE_DATA_DIR`, `ZARR_OUTPUT_PATH`, `START_TIME`, and `END_TIME`.
      * Review `source_configs` (a list of dictionaries) for each radar product's path, interval, processing flags (`to_rainrate`, `antialias`, `compute_clutter`, `create_timemask`, `create_max_intensity`, `pad_value`), and variable definitions.
2.  **Run the script**:
    ```bash
    python scripts/radar.py
    ```

### Satellite Level 1.5 Data Processing (`sat_l1p5.py`)

This script processes Level 1.5 satellite imagery (Infrared and Visible bands). It reads NetCDF files from `.tar` archives, performs spatial regridding to a predefined target grid, and saves the transformed data to Zarr.

1.  **Configure `sat_l1p5.py`**:
      * Open `scripts/sat_l1p5.py`.
      * Adjust `BASE_DATA_DIR`, `ZARR_OUTPUT_PATH`, `START_TIME`, and `END_TIME`.
      * Review `source_config` for input path, interval, fill value, and variable definitions.
      * **Crucially**: Verify `BBOX` and the implicit (or explicit if present in full code) `RadarGridConverter` parameters (projection string, target grid dimensions, corner coordinates) which are vital for accurate regridding. These must match your intended output grid.
2.  **Run the script**:
    ```bash
    python scripts/sat_l1p5.py
    ```

### Satellite Level 2 Data Processing (`sat_l2.py`)

This script processes Level 2 satellite products (e.g., cloud masks, optical thickness, precipitation). It reads NetCDF files from `.tar` archives, regrids them to a target grid, and stores them in the Zarr format.

1.  **Configure `sat_l2.py`**:
      * Open `scripts/sat_l2.py`.
      * Adjust `BASE_DATA_DIR`, `ZARR_OUTPUT_PATH`, `START_TIME`, and `END_TIME`.
      * Review `source_config` for input path, interval, fill value, and variable definitions.
      * **Crucially**: Verify `BBOX` and the implicit (or explicit if present in full code) `RadarGridConverter` parameters for accurate regridding, similar to `sat_l1p5.py`.
2.  **Run the script**:
    ```bash
    python scripts/sat_l2.py
    ```

## 5\. Output Dataset Structure (Zarr)

The result of running these scripts is a single, consolidated Zarr dataset (e.g., `dataset.zarr` or `dataset_regrid.zarr`), located at the `ZARR_OUTPUT_PATH` you configure. This Zarr store is an N-dimensional array store designed for concurrent read/write access and cloud-native computing.

The structure within the Zarr group will typically include:

  * **Top-level Groups**: Each script creates its own subgroup (e.g., `/aws`, `/harmonie`, `/radar`, `/satellite/L_1p5`, `/satellite/L_2`) within the main Zarr store.
  * **Data Arrays**: Within each subgroup, individual environmental variables are stored as Zarr arrays (e.g., `/aws/TOR_RI_PWS_10`, `/harmonie/SHTFL_GDS0_HTGL`, `/radar/rtcor`, `/satellite/L_1p5/IR_016`, `/satellite/L_2/cldmask`).
  * **Dimensions**: Arrays are consistently indexed by `time` (or `time`, `forecast_len` for HARMONIE), and spatial dimensions (e.g., `num_stations` for AWS, or `height`, `width` for gridded data like Radar, Satellite, and HARMONIE).
  * **Coordinate Arrays**: `lat` and `lon` arrays are created within relevant subgroups (e.g., for HARMONIE and Satellite data) to provide geographical context for the gridded data.
  * **Time Mask**: A `time_mask` array might be created at the group level to indicate the presence or absence of data for each timestamp.
  * **Metadata (Attributes)**: Each Zarr group and array will have associated attributes (`attrs`) containing important metadata such as:
      * `start_time`, `end_time`, `interval_minutes`: Defining the temporal coverage and resolution.
      * `fill_value`: The value used to represent missing data.
      * `creation_date`: Timestamp of when the data was processed.
      * `bbox`: Bounding box of the spatial data.
      * Product-specific flags: e.g., `to_rainrate`, `antialias`, `compute_clutter` for radar data.
      * `product`: Original product name (e.g., `rtcor`).
      * `pad_value`: Value used for padding in regridding.

## 6\. Configuration

All configuration is currently handled via variables defined at the top of each Python script. For usage, you will need to directly edit these files.

Key configurable parameters across the scripts:

  * `BASE_DATA_DIR`: The root directory where all your raw input data is located. **This is critical and must be set correctly for each script.**
  * `ZARR_OUTPUT_PATH`: The full path to the Zarr dataset you are creating or appending to.
  * `START_TIME`, `END_TIME`: Pandas Timestamps defining the start and end of the desired data processing period.
  * `source_config` (or `source_configs` in `radar.py`): A dictionary (or list of dictionaries) that specifies details for each data product, including:
      * `path`: The relative path from `BASE_DATA_DIR` to the specific input data for that product.
      * `interval`: The time resolution of the data in minutes.
      * `fill_value`: The value used for missing data.
      * `chunk_size`, `shard_size`: Zarr optimization parameters for how data is stored and accessed.
      * `vars`: A dictionary defining variables to process, their NumPy `dtype`, and `dims` (the shape of the variable's array excluding the time dimension).
      * Additional product-specific flags/parameters: e.g., `to_rainrate`, `antialias`, `compute_clutter`, `forecast_range`, `deacc`, `create_timemask`, `create_max_intensity`, `pad_value`, `product` etc.
  * `BBOX`: A tuple defining the bounding box for spatial data, used in regridding.
  * **Regridding Parameters (Implicit/Explicit `RadarGridConverter` settings)**: For `harmonie.py`, `radar.py`, `sat_l1p5.py`, and `sat_l2.py`, the scripts rely on internal or conceptual grid definitions. These include:
      * The `pyproj` `proj_str` (e.g., `"+proj=stere +lat_0=90 +lon_0=0 +lat_ts=60 +a=6378.14 +b=6356.75 +x_0=0 +y_0=0"`).
      * The target grid dimensions (`original_cols`, `original_rows`).
      * Padding values (`pad_to_divisible`).
      * The projected coordinates of the grid corners (`x_ul`, `y_ul`, `x_lr_original`, `y_lr_original`).
        **These parameters are crucial for correct spatial alignment and must precisely match the target grid definition.**

## 7\. Assumptions

  * **Data Availability**: It is assumed that the raw input data files (e.g., Parquet, NetCDF, HDF5 within TARs) are available at the paths specified in the scripts.
  * **File Naming Conventions**: The scripts rely on specific file naming patterns to extract timestamps and identify data types. Any deviation in the source data's naming convention may lead to files being skipped or misprocessed.
  * **Data Consistency**: Assumes that the internal structure of NetCDF/HDF5 files (variable names, dimensions) matches what the scripts expect.
  * **Time Alignment**: While each script processes its own data type, the intention is to build a unified dataset. Consistent `START_TIME`, `END_TIME`, and `interval` settings across scripts are important for a coherent output Zarr.
  * **Spatial Grid Definition**: The projection and grid parameters embedded or conceptually used in the scripts (especially for HARMONIE, Radar, Satellite) are *critically assumed to match the desired target grid*. Inaccuracies here will result in spatial misalignments in the final Zarr dataset.
  * **Resource Availability**: Processing large datasets can be memory and CPU intensive. The scripts assume sufficient system resources are available.