---

# Config File Documentation

This folder contains YAML configuration files used for different model setups, training procedures, and evaluations in the nowcasting framework. Each config file consists of several sections such as `dataloader`, `model`, `trainer`, `optimizer`, and `eval`. This document provides a general explanation of the structure, common components, available options, and typical defaults for these configuration files.

---

## Config Overview

---

### `dataloader`

Defines how data is loaded and preprocessed for training, validation, and testing. The training and evaluation pipeline uses a PyTorch Lightning `DataModule` by default, located in `nowcasting/data/dataloader_zarrv3.py`. This dataloader works for sequence to sequence timeseries data, where mutliple variables can be selected, grouped and transfomred.

#### Parameters

* **`module_path`**
  Python module for the dataloader class.
  *Example*: `nowcasting.data.dataloader_zarrv3.NowcastingDataModule`

* **`path`**
  Path to the input Zarr dataset.
  *Type*: `str`

* **`var_info`**
  Variables to load from the Zarr dataset.

  * **`sample_var`**
    A low-resolution target variable used for efficient sample filtering (e.g., `"radar/max_intensity_grid"`). Should fit in memory. Used to create the sample dataframes ('idx_time', 'idx_height', 'idx_width', 'weight').

  * **`latlon`**
    Whether to include latitude/longitude metadata, as additional channels.
    *Type*: `bool` — *Default*: `false`

  * **`in_vars`**
    List of input variables used by the model.

    Two formats are supported:

    **Flat list** (variables are stacked along the channel dimension):

    ```yaml
    in_vars:
      - "radar/rtcor"
      - "sat_l1p5/WV_062"
      - "sat_l1p5/IR_108"
    ```

    **Nested list** (grouped separately, e.g., for multi-modal input):

    ```yaml
    in_vars:
      - ["radar/rtcor"]
      - ["sat_l1p5/WV_062", "sat_l1p5/IR_108"]
    ```

  * **`out_vars`**
    List of output (target) variables.
    Only **flat lists** are supported:

    ```yaml
    out_vars:
      - "radar/rtcor"
    ```

  * **`transforms`**
    Optional dictionary of data preprocessing transforms to apply to input variables or groups.
    See [Transforms](#transforms) for full syntax and examples.    

    *How it works*:

    * Group-level transforms (e.g., `"radar"`) are applied before variable-level transforms.
    * Each entry is a list of transformation steps.

    *Example*:

    ```yaml
    radar: 
        dbz_normalization: {convert_to_dbz: true}

    sat_l1p5: 
      resize: {scale: 2}
    sat_l1p5/WV_062:
      normalize: {mean: 230.6474, std: 4.9752}
    sat_l1p5/IR_108:
      normalize: {mean: 266.7156, std: 16.5736}
    ```
   
* **`split_info`**

  Defines how to split the dataset into `train`, `val`, and `test` sets, filter missing data, and optionally remove cluttered samples.

  * **`split_rules`**
    Dictionary defining dataset splits by datetime attributes such as `year`, `month`, `weekday`, `hour`, etc.
    These rules are applied in **priority order**, and each timestamp is assigned the first matching split. Any unmatched timestamps are assigned to a default split.

    Each rule is a dictionary where:

    * The **key** is the name of the split (e.g., `"test"`, `"val"`).
    * The **value** is a selector dict that maps datetime attributes to allowed values.
    * Rules are applied **in order** — the **first match wins**.
    * Any timestamps not matching a rule go to a **default split** (usually `"train"`).


    *Supported datetime attributes*:
    * `year`: 4-digit year (e.g., `2023`)
    * `month`: 1–12
    * `day`: 1–31
    * `hour`: 0–23
    * `minute`, `second`: 0–59
    * `weekday`: 0 = Monday, ..., 6 = Sunday

    *Example*:
    
    ```yaml
    split_rules:
      test:
        year: [2023]
      val:
        month: [6, 11]
    ```

    * All timestamps from 2023 → `test`
    * Any June or November (in other years) → `val`
    * Everything else → `train` (default)
  
    *More examples*:
  
    ```yaml
    split_rules:
      test:
        weekday: [5, 6]  # Weekends
      val:
        weekday: [0, 1, 2, 3, 4] # Weekdays
    ```

    ```yaml
    split_rules:
      test: # Summer afternoons
        month: [6, 7, 8]
        hour: [12, 13, 14, 15]
    ```

    You can combine rules using multiple keys. The split assignment function checks each timestamp in order and assigns it to the **first matching** split.


  * **`apply_missing_masks`**
    Filters out samples that have missing data in any required input group.
    Each group must define a `time_mask` vector that indicates missing data (usually per timestep). These masks are aligned across time resolution and merged to exclude incomplete samples.

    *Example*:

    ```yaml
    apply_missing_masks:
      - radar
      - harmonie
    ```

  * **`clutter_threshold`**
    Removes samples that contain excessive **clutter**, which is detected using a gradient-based method.
    A pixel is marked as clutter if its gradient exceeds **30 mm/h**. If the number of clutter pixels in a sample exceeds this threshold, the sample is excluded.

    *Type*: `int`
    *Default*: `50` (i.e., samples with more than 50 clutter pixels are removed)

* **`sample_info`**
  Sampling rules used to select training and validation samples.

  * **`methods`**

    * `agg`: Aggregation method (e.g., `"mean_pool" or "max_pool"`)
    * `center_crop`: Whether to center-crop the image for evaluation
   
  * **`threshold`**
    Minimum intensity sample weight threshold to consider. The sample weight is computed by the aggregation method (`agg`).

* **`context_len`**
  Number of historical timesteps used as input.
  *Type*: `int`
  *Default*: `4`

* **`forecast_len`**
  Number of future timesteps to predict.
  *Type*: `int`
  *Default*: `18`

* **`img_size`**
  Spatial resolution of of the image for the `sample_var`. This means that the true resolution is `img_size` multipied by 32 (32*[8, 8]=[256, 256]).
  *Type*: `List[int]`
  *Default*: `[8, 8]`

* **`stride`**
  How the window moves across time, height and width.
  *Type*: `List[int]`
  *Default*: `[1, 1, 1]`

* **`include_timestamps`**
  Weather to include timestamp labels for the input sequence. If context length is four the timestamps would be [-4, -3, -2, -1].
  *Type*: `bool`
  *Default*: `False`

* **`batch_size`**
  Number of samples per batch.
  *Type*: `int`
  *Default*: `16`

* **`num_workers`**
  Number of workers for the DataLoader.
  *Type*: `int`
  *Default*: `8`

---

### `model`

Defines which model and model parameterrs should be used. The specific parameters are unique to the selected model. For more info on the model parameters look (here).

#### Parameters 

* **`module_path`**
  Points to the model implementation.
  
  *Examples*:
  
  * `nowcasting.models.ldcast.models.nowcast.nowcast.AFNONowcastModule`
  * `nowcasting.models.earthformer.earthformer.CuboidSEVIRPLModule`
 
* **`checkpoint_path`**
  Optional checkpoint path to load pretrained network. Checkpoint file has to match the model parameters.
  *Type*: `str`
  *Default*: `null`

* **`checkpoint_path`**
  Optional checkpoint path to load pretrained network. Checkpoint file has to match the model parameters.
  *Type*: `str`
  *Default*: `null`

* **`compile`**
  Use torch's `torch.compile`. This should speed-up most pytorch model 1.5-2x.
  *Type*: `bool`
  *Default*: `False`

  ...

---

### `trainer`

Controls the training loop and hardware settings. These parameters are to load the Lightning trainer. 

#### Parameters

* **`max_epochs`**
  Total number of training epochs.

* **`accelerator`**
  Hardware backend: `"auto"`, `"gpu"`, or `"cpu"`

* **`strategy`**
  Distributed training strategy (e.g., `"ddp"`, `"ddp_find_unused_parameters_true"`)

* **`precision`**
  Training precision, e.g., `"16-mixed"` for mixed-precision

* **`profiler`**
  Profiler type for performance analysis

* **`gradient_clip_val`**, **`accumulate_grad_batches`**
  Controls for training stability

* **`limit_train_batches`**
  Useful for debugging or short test runs

---

### `logger`

Settings for experiment logging (e.g., TensorBoard)

#### Parameters

* **`save_dir`**
  Base directory for saved logs

* **`name`**
  Name of the experiment (used in folder structure)

---

### `optimizer`

Training optimization algorithm and scheduler. 

#### Common Fields

* **`method`**
  Optimizer type (e.g., `"adamw"`)

* **`lr`**
  Learning rate

* **`weight_decay`**
  L2 weight regularization

* **`betas`**
  Parameters for Adam optimizer

* **`scheduler`**
  Learning rate scheduler settings:

  * `patience`, `factor`: used for ReduceLROnPlateau
  * `warmup_percentage`: fraction of training steps for linear warm-up

---

### `loss`

Loss function configuration.

#### Parameters

* **`loss_name`**
  Type of loss: `"mae"`, `"mse"`, `"balanced"`, `"multi-source"`, etc.

* **`min_val`**, **`max_val`** (`"balanced"` and `"multi-source"`)
  Normalization bounds used in loss scaling

* **`weight_intensity`**, **`max_weight_r`** (`"balanced"` and `"multi-source"`)
  Adjustments for radar intensity class balancing

* **`extended`** (`"balanced"` and `"multi-source"`)
  Whether to use extended/balanced formulation

* **`channel_weights`** (`"multi-source"`)
  Per-channel weights in multi-source loss

---

### `callbacks`

Training callbacks for control and checkpointing.

### Options

* **`checkpoint`**

  * `monitor`: Metric to track (e.g., `"val/loss"`)
  * `save_top_k`: Number of best models to keep
  * `save_last`: Save the final model at the last epoch

* **`early_stopping`** (optional)

  * Stops training if no improvement after N epochs

---

### `eval`

Configuration for evaluation runs.

#### Parameters

* **`model_path`**
  Path to evaluation wrapper

* **`model_name`**
  Short model identifier used in logs and filenames

* **`eval_name`**
  Dataset split to evaluate (`"val"` or `"test"`)

* **`leadtimes`**
  Forecast lead times (in minutes)

* **`thresholds`**
  Detection metric thresholds (e.g., for FSS, CSI)

---

### Notes

* Variable names like `radar/rtcor` or `sat_l1p5/WV_062` must match those in the input Zarr dataset.
* Use `img_size`, `context_len`, `forecast_len`, and `stride` to control temporal and spatial resolution.
* Mixed inputs (radar + satellite) are supported with multi-modal autoencoder pipelines.
* Autoencoders can be pretrained and loaded using `checkpoint_path`.

---

<a id="transforms"></a>
## Transforms

Here are the available transforms and their configuration options.

### `normalize`
Performs standard Z-score normalization (`(data - mean) / std`).

- **`mean`**: (`float` | `list`) The mean value(s).
- **`std`**: (`float` | `list`) The standard deviation value(s).
- **`aws`**: (`bool`, optional) Set to `true` if normalizing AWS data, which has a different shape. Default: `false`.

*Example (single channel):*
```yaml
transforms:
  sat_l1p5/IR_108:
    normalize: {mean: 266.7156, std: 16.5736}
```
*Example (group with multiple channels):*
```yaml
transforms:
  sat_l1p5:
    normalize:
      mean: [230.6474, 266.7156]
      std: [4.9752, 16.5736]
```

---

### `dbz_normalization`
A specialized transform for radar data that converts rain rates (mm/h) to dBZ, applies normalization, and clips the values.

- **`convert_to_dbz`**: (`bool`, optional) Whether to apply the dBZ conversion. Default: `true`.
- **`norm_method`**: (`string`, optional) Normalization method.
  - Options: `'minmax'`, `'minmax_tanh'`, `null`.
  - Default: `'minmax'`.

*Example:*
```yaml
transforms:
  radar/rtcor:
    dbz_normalization:
      convert_to_dbz: true
      norm_method: 'minmax'
```

---

### `resize`
Resizes image data using OpenCV. Specify either `target_shape` or `scale`.

- **`target_shape`**: (`list`) The target output shape `[height, width]`.
- **`scale`**: (`float`) A scaling factor to apply to both height and width.
- **`method`**: (`string`, optional) Interpolation method.
  - Options: `'nearest'`, `'bilinear'`, `'area'`, `'lanczos'`.
  - Default: `'nearest'`.

*Example (by scale factor):*
```yaml
transforms:
  sat_l1p5:
    resize:
      scale: 0.5
      method: 'bilinear'
```
*Example (by target shape):*
```yaml
transforms:
  radar/rtcor:
    resize:
      target_shape: [256, 256]
```

---

### `default_rainrate`
A transform for rain rate data that applies a threshold, logarithmic scaling, and finally Z-score normalization.

- **`threshold`**: (`float`, optional) Values below this are set to `fill_value`. Default: `0.1`.
- **`fill_value`**: (`float`, optional) The value used for data below the threshold. Default: `0.02`.
- **`mean`**: (`float`, optional) Mean value for Z-score normalization. Default: `0`.
- **`std`**: (`float`, optional) Standard deviation for Z-score normalization. Default: `1`.

*Example:*
```yaml
transforms:
  radar/rtcor:
    default_rainrate:
      threshold: 0.05
      mean: -1.2
      std: 0.9
```

---

### `aws_interpolation`
Interpolates sparse Automatic Weather Station (AWS) data onto a dense grid. This transform expects the input data to be a tuple of `(values, locations)`.

- **`target_shape`**: (`list`) The target output shape `[height, width]`.
- **`method`**: (`string`, optional) Interpolation method.
  - `'dense'`: Inverse Distance Weighting (IDW).
  - `'sparse'`: Draws a disk with the station's value at its location.
  - Default: `'dense'`.
- **`radius`**: (`int`, optional) Disk radius for the `'sparse'` method. Default: `3`.
- **`power`**: (`int`, optional) Power for IDW interpolation in the `'dense'` method. Default: `2`.
- **`search_radius`**: (`int`, optional) Number of nearest neighbors for IDW in the `'dense'` method. Default: `5`.
- **`downscale`**: (`int`, optional) Factor by which to downscale the grid before `'dense'` interpolation for efficiency. Default: `8`.

*Example:*
```yaml
transforms:
  aws/10min_mean_val:
    aws_interpolation:
      target_shape: [768, 704]
      method: 'dense'
      search_radius: 10
```

---

### `regrid`
Regrids data from a source projection to a target grid using nearest-neighbor interpolation. This is useful for aligning data from different sources, like weather models (e.g., HARMONIE).

- **`src_path`**: (`string`) Path to the source Zarr dataset containing the source grid's latitude and longitude arrays.
- **`lat_name`**: (`string`, optional) Name of the latitude variable in the source file. Default: `'lat'`.
- **`lon_name`**: (`string`, optional) Name of the longitude variable in the source file. Default: `'lon'`.
- **`downscale_factor`**: (`int`, optional) Factor to downscale the target grid. Default: `1`.

*Example:*
```yaml
transforms:
  harmonie/ta2m:
    regrid:
      src_path: "/path/to/harmonie_grid.zarr"
      lat_name: "latitude"
      lon_name: "longitude"
```

---

### `dropout`
A data augmentation technique that sets the entire input array to zero with a given probability.

- **`p`**: (`float`, optional) The probability of setting the input to zero. Default: `0.5`.

*Example:*
```yaml
transforms:
  radar/rtcor:
    dropout:
      p: 0.05
```

---
