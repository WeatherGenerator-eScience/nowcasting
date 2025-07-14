import zarr
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import pytorch_lightning as pl
from skimage.draw import disk
from scipy.spatial import KDTree
from scipy.ndimage import zoom

from typing import Optional, List, Tuple
import re

import os
import numba as nb
import torch

# Configure threading layer to avoid TBB warnings
os.environ["NUMBA_THREADING_LAYER"] = "omp"

@nb.njit(parallel=True, fastmath=True, nogil=True)
def spatial_max_pool(input_data, output, kh, kw, sh, sw):
    """Parallel 2D spatial max pooling with configurable kernel and stride"""
    n_times, in_h, in_w = input_data.shape
    out_h = (in_h - kh) // sh + 1
    out_w = (in_w - kw) // sw + 1
    
    for t in nb.prange(n_times):
        for i in range(out_h):
            h_start = i * sh
            for j in range(out_w):
                w_start = j * sw
                max_val = 0
                # Optimized inner loops
                for di in range(kh):
                    for dj in range(kw):
                        val = input_data[t, h_start+di, w_start+dj]
                        max_val = max(max_val, val)
                output[t, i, j] = max_val

@nb.njit(parallel=True, fastmath=True, nogil=True)
def temporal_max_pool(input_data, output, kt, st):
    """Parallel 1D temporal max pooling with configurable kernel and stride"""
    n_times, h, w = input_data.shape
    out_t = (n_times - kt) // st + 1
    
    for spatial_idx in nb.prange(h * w):
        # Convert flat index to h,w coordinates
        h_idx = spatial_idx // w
        w_idx = spatial_idx % w
        for t_out in range(out_t):
            t_start = t_out * st
            max_val = 0
            # Optimized temporal window
            for dt in range(kt):
                val = input_data[t_start + dt, h_idx, w_idx]
                max_val = max(max_val, val)
            output[t_out, h_idx, w_idx] = max_val

def max_pool_3d(input_array, kernel_size, stride=(1, 1, 1)):
    """
    Efficient 3D max pooling using separate spatial/temporal steps
    Args:
        input_array: 3D array (time, height, width)
        kernel_size: (temporal_k, height_k, width_k)
        stride: (temporal_stride, height_stride, width_stride)
    Returns:
        Pooled 3D array with reduced dimensions
    """
    # Validate inputs
    if input_array.ndim != 3:
        raise ValueError("Input must be 3D array (time, height, width)")
    kt, kh, kw = kernel_size
    st, sh, sw = stride
    n_times, in_h, in_w = input_array.shape

    # Validate dimensions
    if kh > in_h or kw > in_w or kt > n_times:
        raise ValueError("Kernel size exceeds input dimensions")
    if any(s < 1 for s in (kt, kh, kw, st, sh, sw)):
        raise ValueError("Kernel and stride values must be ≥1")

    # Spatial pooling first
    out_h = (in_h - kh) // sh + 1
    out_w = (in_w - kw) // sw + 1
    spatial_pooled = np.empty((n_times, out_h, out_w), dtype=input_array.dtype)
    spatial_max_pool(input_array, spatial_pooled, kh, kw, sh, sw)

    # Temporal pooling second
    out_t = (n_times - kt) // st + 1
    final_output = np.empty((out_t, out_h, out_w), dtype=input_array.dtype)
    temporal_max_pool(spatial_pooled, final_output, kt, st)

    return final_output

class IndexManager:
    def __init__(self, 
        sequence_len=22, 
        pool_kernel=(22, 8, 8), 
        pool_stride=(1, 1, 1)
        ):
        self.sequence_len = sequence_len
        self.pool_kernel = pool_kernel
        self.pool_stride = pool_stride
        # set when splitting
        self.indices = None              # global time indices matching regex
        self.valid_relative = None       # positions in filtered time array
        # set when spatial dims known
        self.Hp = None
        self.Wp = None
        self.Tp = None                   # Temporal pooled length (set in get_sampler)
        # sampling flag
        self.sampling = False
        self.valid_pooled_indices = None # Valid indices in pooled time dimension

    def create_split(self, start_time, end_time, interval_minutes, regex):
        timestamps = pd.date_range(start=start_time, end=end_time, freq=f'{interval_minutes}min')
        self.timestamps = timestamps

        pattern = re.compile(regex)
        all_idxs = [i for i, ts in enumerate(timestamps) if pattern.match(ts.strftime('%Y%m'))]
        self.indices = np.array(all_idxs, dtype=int)

        rel = []
        n = len(self.indices)
        for i in range(n - self.sequence_len + 1):
            window = self.indices[i:i + self.sequence_len]
            if window[-1] - window[0] == self.sequence_len - 1:
                rel.append(i)
        self.valid_relative = np.array(rel, dtype=int)

    def prepare_spatial(self, height, width):
        kT, kH, kW = self.pool_kernel
        sT, sH, sW = self.pool_stride
        # spatial pooled dims
        self.Hp = (height - kH) // sH + 1
        self.Wp = (width - kW) // sW + 1

    def total_positions(self):
        """Number of valid temporal windows (no spatial sampling)."""
        return len(self.valid_relative)

    def total_pooled_positions(self):
        """Total positions when sampling over time+space."""
        return self.Tp * self.Hp * self.Wp

    def get_index(self, flat_idx):
        """
        Map a dataset index back to original array indices.
        """
        if self.Hp is None or self.Wp is None:
            self.prepare_spatial(24, 22)  # Default spatial dims if not set

        sT, sH, sW = self.pool_stride

        if not self.sampling:
            t_rel = self.valid_relative[flat_idx]
            t0 = self.indices[t_rel]
            # Random spatial offsets
            # h0 = np.random.randint(0, self.Hp * sH)
            # w0 = np.random.randint(0, self.Wp * sW)
            h0 = (self.Hp * sH)//2
            w0 = (self.Wp * sW)//2
            return (t0, h0, w0)

        # Sampling mode: time × Hp × Wp
        window_idx = flat_idx // (self.Hp * self.Wp)
        rem = flat_idx % (self.Hp * self.Wp)
        h0 = (rem // self.Wp) * sH
        w0 = (rem % self.Wp) * sW

        # Get the valid pooled time index and map to original indices
        t_pool = self.valid_pooled_indices[window_idx]
        t_filtered = t_pool * sT  # Start index in filtered array
        t0 = self.indices[t_filtered]  # Global index in original array
        return (t0, h0, w0)

    def get_length(self):
        return self.total_pooled_positions() if self.sampling else self.total_positions()

    def get_sampler(self, arr, num_samples=None, bins=7, first_bin=0.2, last_bin=15, scaler=255/50):
        _, H, W = arr.shape
        if self.Hp is None or self.Wp is None:
            self.prepare_spatial(H, W)

        self.sampling = True

        filtered = arr[self.indices]
        pooled = max_pool_3d(
            filtered, 
            kernel_size=self.pool_kernel, 
            stride=self.pool_stride)

        # Compute valid pooled time indices
        kT, kH, kW = self.pool_kernel
        sT, sH, sW = self.pool_stride
        valid_pooled_indices = []

        for t_rel in self.valid_relative:
            t_start = t_rel
            t_end = t_rel + self.sequence_len - 1
            # Calculate valid pooled indices for this original valid sequence
            min_t_pool = np.ceil(t_start / sT).astype(int)
            max_t_pool = np.floor((t_end - kT + 1) / sT).astype(int)
            # Clamp to valid range [0, pooled_time_dim - 1]
            max_pooled_time = pooled.shape[0] - 1
            min_t_pool = max(min_t_pool, 0)
            max_t_pool = min(max_t_pool, max_pooled_time)
            if max_t_pool >= min_t_pool:
                valid_pooled_indices.extend(range(min_t_pool, max_t_pool + 1))

        # Deduplicate and store
        self.valid_pooled_indices = np.unique(valid_pooled_indices)
        self.Tp = len(self.valid_pooled_indices)

        valid = pooled[self.valid_pooled_indices]
        flat = valid.ravel()

        # Compute weights based on pooled values
        first_s = first_bin * scaler
        last_s = last_bin * scaler
        edges = np.exp(np.linspace(np.log(first_s), np.log(last_s), bins - 1))
        bin_idx = np.digitize(flat, edges)

        counts = np.bincount(bin_idx, minlength=bins)
        counts = np.where(counts == 0, 1, counts)  # Avoid division by zero
        weights = 1.0 / counts
        w_arr = (weights[bin_idx] * 1e7).astype(np.float32)

        return WeightedRandomSampler(
            weights=torch.from_numpy(w_arr),
            num_samples=num_samples if num_samples else len(w_arr),
            replacement=True
        )

class AWSProcessor:
    def __init__(self, config: dict, grid_width: int, grid_height: int):
        self.method = config.get('method', 'sparse')
        self.config = config

        self.width = grid_width
        self.height = grid_height
        
        if self.method == 'sparse':
            self.radius = config.get('radius', 3)
        elif self.method == 'dense':
            self.downscale = config.get('downscale', 1)
            self.power = config.get('power', 2)
            self.search_radius = config.get('search_radius', 5)
            self.width_down = self.width//self.downscale
            self.height_down = self.height//self.downscale
            self._setup_interpolator()
        else:
            raise ValueError(f"Invalid AWS method: {self.method}")

    def generate_frame(self, x: np.ndarray, y: np.ndarray, values: np.ndarray) -> np.ndarray:
        if self.method == 'sparse':
            return self._generate_sparse(x, y, values)
        elif self.method == 'dense':
            return self._generate_dense(x, y, values)
        else:
            raise NotImplementedError

    def _generate_sparse(self, x, y, values):
        img = np.zeros((self.height, self.width), dtype=np.float32)
        for xi, yi, val in zip(x, y, values):
            if 0 <= yi < img.shape[0] and 0 <= xi < img.shape[1]:
                rr, cc = disk((yi, xi), self.radius, shape=img.shape)
                img[rr, cc] = val
        return img

    def _setup_interpolator(self):
        self.x_grid, self.y_grid = np.meshgrid(
            np.linspace(0, self.width_down - 1, self.width_down),
            np.linspace(0, self.height_down - 1, self.height_down)
        )

        self.grid_points = np.column_stack((self.x_grid.ravel(), self.y_grid.ravel()))

    def _interpolate(self, stations):
        """Interpolate using inverse distance weighting"""

        k = min(self.search_radius, len(stations))
        tree = KDTree(stations[:, :2])
        distances, indices = tree.query(self.grid_points, k=k)

        if k == 1:
            distances = distances[:, None]
            indices = indices[:, None]

        weights = 1 / ((distances ** self.power) + 1)
        values = stations[indices, 2]
        weighted_sums = np.sum(values * weights, axis=1)
        sum_weights = np.sum(weights, axis=1)

        return weighted_sums / sum_weights

    def _generate_dense(self, x, y, values):
        if len(x) == 0:
            return np.zeros((self.height, self.width), dtype=np.float32)
        
        stations = np.column_stack((x, y, values))
        stations[:, :2] //= self.downscale

        idw_values = self._interpolate(stations)
        idw_grid = idw_values.reshape(self.height_down, self.width_down)

        if self.downscale>1:
            return zoom(idw_grid, self.downscale, order=0)

        return idw_grid.astype(np.float32)

class NowcastingDataset(Dataset):
    _aws_global_cache = {}

    def __init__(self,
                 root,
                 info: dict,
                 im: IndexManager,
                 context_len: int = 4,
                 forecast_len: int = 18,
                 img_size: tuple[int,int] = (8,8),
                 dtype = np.float32,
                 ldcast = False):
        self.root = root
        self.info = info
        self.im = im
        self.context_len = context_len
        self.forecast_len  = forecast_len
        self.img_h, self.img_w = img_size
        self.dtype = dtype
        self.ldcast = ldcast
        self.block_size = 32
        self.aws_cache = {}

        # first var defines (T, H, W)
        main_var = info['main_var']
        _, H, W = self.root[main_var].shape
        self.H, self.W = H, W

        # Preload sparse AWS data
        for var, config in self.info['vars'].items():
            if 'aws' in var:
                self._preload_aws(var, config)

    def _preload_aws(self, var: str, config: dict):
        """Load AWS data with configurable processing method."""
        if var in NowcastingDataset._aws_global_cache:
            # Use existing entry from global cache
            self.aws_cache[var] = NowcastingDataset._aws_global_cache[var]
            return

        # Load data if not in cache
        var_name = var.split('/')[-1]
        path = f"/vol/csedu-nobackup/project/mrobben/nowcasting/dataset/aws/xy/{var_name}.parquet"
        df = pd.read_parquet(path)

        # Store raw station data per timestamp
        cache = {}
        for timestamp, group in df.groupby('T'):
            cache[timestamp] = (
                group['X'].values.astype(int),
                group['Y'].values.astype(int),
                group['value'].values.astype(self.dtype)
            )

        cache['mean'] = df['value'].mean()
        cache['std'] = df['value'].std()
        del df

        # Initialize processor with current grid dimensions
        processor = AWSProcessor(
            config=config,
            grid_width=self.W,
            grid_height=self.H
        )

        NowcastingDataset._aws_global_cache[var] = {
            'cache': cache,
            'processor': processor
        }
        self.aws_cache[var] = NowcastingDataset._aws_global_cache[var]

    def _generate_aws_frame(self, var: str, timestamp: int) -> np.ndarray:
        """Generate AWS frame using configured method."""
        cache_info = self.aws_cache[var]
        x, y, values = cache_info['cache'].get(timestamp, (np.array([]), np.array([]), np.array([])))
        img = cache_info['processor'].generate_frame(x, y, values)
        img -= cache_info['cache'].get('mean', 0)
        img /= cache_info['cache'].get('std', 0)
        return img

    def __len__(self):
        return self.im.total_positions()

    def _process_radar(self, img, mean=-0.051, std=0.528):
        img[img < 0.1] = 0.02
        np.log10(img, out=img)

        img -= mean
        img /= std

    def __getitem__(self, idx):

        t0, h0, w0 = self.im.get_index(idx)

        # Compute random crop once
        h0 = h0 * self.block_size
        w0 = w0 * self.block_size
        h1 = h0 + self.img_h*self.block_size
        w1 = w0 + self.img_w*self.block_size

        # time slice covering both context & future
        t1 = t0 + self.context_len
        t2 = t1 + self.forecast_len

        context = []
        future = []

        main_var = self.info['main_var']
        dropout_cfg = self.info.get('dropout', {})

        # Decide dropout for main_var
        main_var_key = main_var.split('/')[0]
        if np.random.rand() < dropout_cfg.get(main_var_key, 0.0):
            block = np.zeros((t2 - t0, h1 - h0, w1 - w0), dtype=self.dtype)
        else:
            block = self.root[main_var][t0:t2, h0:h1, w0:w1].astype(self.dtype)
            # data_prep(block, convert_to_dbz=True, norm_method="minmax")
            self._process_radar(block)

        context.append(block[:self.context_len])
        future.append(block[self.context_len:])

        for var, config in self.info['vars'].items():
            t_scale = config['t_scale']
            t0_low = int(t0/t_scale + 0.5)
            t1_low = int(t1/t_scale + 0.5)

            var_key = var.split('/')[0]
            if np.random.rand() < dropout_cfg.get(var_key, 0.0):
                blk_up = np.zeros((self.context_len, h1 - h0, w1 - w0), dtype=self.dtype)
            else:
                if 'aws' in var:
                    frames = [self._generate_aws_frame(var, t)[h0:h1, w0:w1] for t in range(t0_low, t1_low)]
                    blk_low = np.stack(frames)
                else:
                    blk_low = self.root[var][t0_low:t1_low, h0:h1, w0:w1]
                blk_up = np.repeat(blk_low, repeats=t_scale, axis=0).astype(self.dtype)

            context.append(blk_up.astype(self.dtype))

        context = np.stack(context, axis=0)
        future = np.stack(future, axis=0)

        if self.ldcast:
            timesteps = np.arange(-self.context_len + 1, 1, dtype=np.float32)

            context = [[context[i:i+1], timesteps] for i in range(context.shape[0])]
            
        return context, future

class NowcastingDataModule(pl.LightningDataModule):
    def __init__(
        self,
        ds_path: str,
        var_info: dict,
        context_len: int = 4,
        forecast_len: int = 18,
        img_size = (8, 8),
        train_re: str = '',
        test_re: str = '',
        val_re: str = '',
        ldcast: bool = False,
        batch_size: int = 4,
        num_workers: int = 8,
        num_samples: int = 100_000
        ):
        super().__init__()
        self.ds_path = ds_path
        self.var_info = var_info
        self.context_len = context_len
        self.forecast_len = forecast_len
        self.img_size = img_size
        self.train_re = train_re
        self.test_re = test_re
        self.val_re = val_re
        self.ldcast = ldcast
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.num_samples = num_samples

    def setup(self, stage: Optional[str] = None):
        # Open the zarr dataset
        store = zarr.LMDBStore(
            self.ds_path,
            readonly=True,     # no writers → readers never block
            lock=False,        # don’t take file locks at all
            max_readers=126,   # up the reader limit if you need more simultaneous threads)
        )
        root  = zarr.open(store, mode="r")

        # Create the Index Managers and split the data
        start_time = pd.Timestamp("2020-01-01 00:00")
        end_time = pd.Timestamp("2023-12-31 23:59")
        interval_minutes = 5

        pool_stride = (2,2,2)
        pool_kernel = (self.context_len+self.forecast_len, ) + (self.img_size)

        im_train = IndexManager(sequence_len=self.context_len+self.forecast_len, pool_stride=pool_stride, pool_kernel=pool_kernel)
        im_val = IndexManager(sequence_len=self.context_len+self.forecast_len)
        im_train.create_split(start_time, end_time, interval_minutes, self.train_re)
        im_val.create_split(start_time, end_time, interval_minutes, self.val_re)

        # Create the sampler
        self.train_sampler = im_train.get_sampler(root[self.var_info['sample_var']], num_samples=self.num_samples)

        # Initialize the datasets
        self.train_dataset = NowcastingDataset(
            root=root,
            info=self.var_info,
            im=im_train,
            context_len=self.context_len,
            forecast_len=self.forecast_len,
            img_size=self.img_size,
            ldcast=self.ldcast
        )
        self.val_dataset = NowcastingDataset(
            root=root,
            info=self.var_info,
            im=im_val,
            context_len=self.context_len,
            forecast_len=self.forecast_len,
            img_size=self.img_size,
            ldcast=self.ldcast
        )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            sampler=self.train_sampler,
            pin_memory=True,
            persistent_workers=self.num_workers > 0,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
            pin_memory=True,
            persistent_workers=self.num_workers > 0
        )

if __name__ == "__main__":
    # 1) open LMDB-backed Zarr
    path = "/vol/csedu-nobackup/project/mrobben/nowcasting/dataset/radar.lmdb"
    var_info = {
        "sample_var": "radar/rtcor/max_intensity_grid",
        "main_var": "radar/rtcor/radar_image",
        "dropout": {"radar": 0.0, "aws": 0.0},
        "vars": {
            # "aws/sparse/TOT_T_DRYB_10": {'t_scale': 2, 'method': 'sparse', 'radius': 4},
            "aws/dense/TOT_T_DRYB_10": {'t_scale': 2, 'method': 'dense', 'power': 2, 'search_radius': 5, 'downscale': 8},
            "aws/dense/TOT_T_DEWP_10": {'t_scale': 2, 'method': 'dense', 'power': 2, 'search_radius': 5, 'downscale': 8},
            "aws/dense/TOT_T_WETB_10": {'t_scale': 2, 'method': 'dense', 'power': 2, 'search_radius': 5, 'downscale': 8},
            "aws/dense/TOW_FF_10M_10": {'t_scale': 2, 'method': 'dense', 'power': 2, 'search_radius': 5, 'downscale': 8},
            "aws/dense/TOA_P_STN_LEVEL_10": {'t_scale': 2, 'method': 'dense', 'power': 2, 'search_radius': 5, 'downscale': 8},
        }
    }

    train_re=r'20(20|21|22)(01|02|03|04|05|07|08|09|10|12)'  # 2020-2022, non-June/Nov months
    val_re=r'20(20|21|22)(06|11)'                            # May/Nov 2020-2022
    test_re=r'2023\d{2}'                                     # All of 2023
    
    data_module = NowcastingDataModule(
        path,
        var_info,
        context_len = 4,
        forecast_len = 18,
        train_re=train_re,
        val_re=val_re,
        test_re=test_re,
        batch_size = 8,
        num_workers = 16,
    )
    data_module.setup()

    # 2) timing test
    import time
    start = time.time()
    for i, (context, future) in enumerate(data_module.train_dataloader()):
        pass
        print(context.shape, future.shape)
        if i == 100:
            break
    print("Elapsed:", time.time() - start)
