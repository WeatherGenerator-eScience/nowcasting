# AWS Data Processing Pipeline

This folder contains a set of Bash and Python scripts to download, process, and transform Automatic Weather Station (AWS) data from the KNMI RobuKIS platform. The pipeline handles data acquisition, cleaning, conversion to optimized Parquet format, and transformation of geographic coordinates (latitude, longitude) into a projected grid (X, Y) with a unified time index (T).

## 1. Overview

This pipeline consists of three main components:

* **`downloadAWS.sh`**: A Bash script to download raw AWS data in CSV format from the KNMI RobuKIS service. It divides large downloads into smaller chunks for improved reliability.
* **`process_aws.py`**: A Python script that takes the raw CSV files, performs data cleaning (e.g., fixing timestamps, filtering stations), and converts them into optimized Parquet files. Each measurement variable is stored in a separate Parquet file.
* **`aws_latlon_to_xy.py`**: A Python script that transforms the processed Parquet files. It converts geographic latitude and longitude coordinates into X, Y grid coordinates based on a specified polar stereographic projection and maps timestamps to a sequential time index (T). This output is suitable for gridded analysis or machine learning applications.

## 2. Prerequisites

Before you begin, ensure you have the following installed:

* **Bash:** Standard on most Linux/macOS systems.
* **`wget`:** A command-line utility for downloading files.
    * On Debian/Ubuntu: `sudo apt-get install wget`
    * On macOS (with Homebrew): `brew install wget`
* **Python 3.8+:**
* **Required Python Libraries:** Required Python Libraries: These are essential for running the scripts. It is recommended to install them within a virtual environment.
      * `pandas`
      * `numpy`
      * `fastparquet`
      * `pyproj`
      * `tqdm`

## 3. Repository Structure

```
process_aws/
├── downloadAWS.sh
├── process\_aws.py
├── aws\_latlon\_to\_xy.py
├── README.md
└── requirements.txt
````

## 4. Usage

Follow these steps sequentially to run the entire data pipeline.

### Step 1: Generate RobuKIS Query File (`table.in`)

The `downloadAWS.sh` script requires an input file named `table.in` which defines the specific AWS data parameters to be downloaded from the KNMI RobuKIS platform.

1.  Go to the RobuKIS data selection interface: `https://robukis-core.widi.knmi.cloud/servlet/selection/SelectTableElementGroup/`
2.  **Configure your query:**
    * Select the desired AWS stations (e.g., all Dutch stations).
    * Choose the measurement parameters (e.g., temperature, wind speed, precipitation).
    * Set the desired time resolution (e.g., 10 minutes).
    * *(Optional: You can find the original query content for `table_10_mindata_zonder_luchtvochtigheid.in` if you need a starting point for data without humidity).*
3.  Scroll to the very bottom of the page and click the "**Show recipe**" button.
4.  A text area will appear with the "Recipe" (XML/form data). Copy this entire content.
5.  Create a file named `table_10_mindata_zonder_luchtvochtigheid.in` (or a similar descriptive name) inside the `data/` directory of this repository and paste the copied content into it.
6.  The `downloadAWS.sh` script will automatically copy this to `table.in` before starting downloads.

### Step 2: Download AWS Data (`downloadAWS.sh`)

This script downloads raw CSV data for the years 2020-2023, split by month and further into 5 parts per month for robustness.

1.  **Review the script:**
    * Open `downloadAWS.sh` and verify the `output_dir` variable. By default, it's set to `/nobackup/users/robben/data/AWS/RobuKIS`. Adjust this path if you want to store the raw CSVs elsewhere.
2.  **Run the script:**
    ```bash
    bash downloadAWS.sh
    ```
    This process can take a considerable amount of time depending on the data volume and network speed. Downloaded CSV files will be saved in the `output_dir` you configured.

### Step 3: Process and Convert Data to Parquet (`process_aws.py`)

This Python script cleans the downloaded CSVs and converts them into optimized Parquet files, organized by variable.

1.  **Review and configure the script:**
    * Open `process_aws.py`.
    * **`DATA_DIR`**: **Crucially**, set this variable to the exact path where your CSV files from `downloadAWS.sh` are located.
        * Example: `DATA_DIR = Path("/nobackup/users/robben/data/AWS/RobuKIS")`
    * `OUTPUT_DIR`: This is where the processed Parquet files (with `DS_LAT`, `DS_LON`, `IT_DATETIME`) will be saved. Default is `output/` relative to the script.
    * `TEST_MODE` and `TEST_FILES`: Set `TEST_MODE = True` and `TEST_FILES = N` to process only the first `N` CSV files for quick testing. Remember to set `TEST_MODE = False` for full processing.
    * `COMPRESSION`: Adjust if you prefer a different compression algorithm (e.g., 'SNAPPY', 'ZSTD').
2.  **Run the script:**
    ```bash
    python process_aws.py
    ```
    This will create a subdirectory for each measurement variable within `OUTPUT_DIR` (e.g., `output/TA.parquet`, `output/RH.parquet`).

### Step 4: Transform Lat/Lon to Grid Coordinates (`aws_latlon_to_xy.py`)

This script transforms the geographical coordinates (latitude, longitude) into X, Y grid coordinates and maps timestamps to a sequential time index (T), suitable for gridded applications.

1.  **Review and configure the script:**
    * Open `aws_latlon_to_xy.py`.
    * **`input_dir`**: **Crucially**, set this to the `OUTPUT_DIR` from `process_aws.py` (i.e., where the Parquet files with lat/lon are located).
        * Example: `input_dir = "/nobackup_1/users/robben/full_dataset/aws/latlon"` (This example path should be replaced with your actual path).
    * **`output_dir`**: This is where the final Parquet files (with `X`, `Y`, `T`) will be saved. Default is `/nobackup_1/users/robben/full_dataset/aws/xy` (replace with your desired path).
    * **`RadarGridConverter` parameters**:
        * `original_cols`, `original_rows`: These define the resolution of your target radar grid.
        * `pad_to_divisible`: This padding factor is important for compatibility with certain machine learning frameworks (e.g., for convolutional neural networks). It **must match** the `pad_to_divisible` setting of `AwsImageGenerator` or any other tool consuming this gridded data.
        * `self.proj` (projection string): The `+proj=stere` string and its parameters (e.g., `lat_0`, `lon_0`, `lat_ts`, `a`, `b`, `x_0`, `y_0`) define the **polar stereographic projection**. This string and the associated `x_ul`, `y_ul`, `x_lr_original`, `y_lr_original` coordinates **must exactly match** how your target radar grid's coordinates are defined. Inconsistencies here will lead to incorrect spatial mapping.
    * **`START_TIME`, `END_TIME`, `INTERVAL_MINUTES`**: Adjust these constants to match the desired time range and resolution for your time indexing.
2.  **Run the script:**
    ```bash
    python aws_latlon_to_xy.py
    ```
    A progress bar will show the processing status.

## 6. Output

* **`downloadAWS.sh`**: Raw CSV files named `RobuKIS_table_YYYYMMDD_HHMMSS_YYYYMMDD_HHMMSS.csv` (split into parts) in the configured `output_dir`.
* **`process_aws.py`**: Parquet files, one per measurement variable (e.g., `TA.parquet`, `RH.parquet`), located in the `output/` directory (or your specified `OUTPUT_DIR`). Each file contains `StationID`, `DS_LAT`, `DS_LON`, `IT_DATETIME`, and `value`.
* **`aws_latlon_to_xy.py`**: Transformed Parquet files, mirroring the variable names from the previous step, located in the `xy_output/` directory (or your specified `output_dir`). Each file contains `X` (grid column), `Y` (grid row), `T` (time index), and `value`.

## 7. Configuration

Currently, configurations are hardcoded within the Python scripts. For future improvements and easier management, consider externalizing these into a `config/settings.yaml` file and using a library like `PyYAML` to load them.

Key configurable parameters:

* **`downloadAWS.sh`**:
    * `output_dir`
    * `year` and `month` ranges
* **`process_aws.py`**:
    * `DATA_DIR` (input CSV path)
    * `OUTPUT_DIR` (output Parquet path for lat/lon data)
    * `TEST_MODE`, `TEST_FILES`
    * `COMPRESSION`, `ROW_GROUP_SIZE`
    * `PREFIX_COLS` (expected fixed columns from RobuKIS)
* **`aws_latlon_to_xy.py`**:
    * `input_dir` (input Parquet path from `process_aws.py`)
    * `output_dir` (output Parquet path for X,Y,T data)
    * `START_TIME`, `END_TIME`, `INTERVAL_MINUTES`
    * `RadarGridConverter` parameters: `original_cols`, `original_rows`, `pad_to_divisible`, projection parameters (`+proj=stere`, `lat_0`, `lon_0`, `lat_ts`, `a`, `b`, `x_0`, `y_0`), and corner coordinates (`x_ul`, `y_ul`, `x_lr_original`, `y_lr_original`).