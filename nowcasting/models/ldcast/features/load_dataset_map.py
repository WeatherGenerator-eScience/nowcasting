from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import torch
import pytorch_lightning as pl

import xarray as xr
import numpy as np
import pandas as pd

from typing import Optional, List, Tuple

import time

def split_df_by_months(
    df: pd.DataFrame, 
    val_months: List[str], 
    test_months: List[str]
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    # Create a copy to avoid modifying the original dataframe.
    df = df.copy()
    
    # Extract the month part (first 6 characters) of the timestamp.
    df['month'] = df['timestamp'].str[:6]
    
    # Select rows for validation and test based on the month.
    val_df = df[df['month'].isin(val_months)]
    test_df = df[df['month'].isin(test_months)]
    
    # The training set is everything that is not in validation or test.
    train_df = df[~df['month'].isin(val_months + test_months)]
    
    # Optionally, drop the extra 'month' column if you don't need it further.
    train_df = train_df.drop(columns=['month'])
    val_df = val_df.drop(columns=['month'])
    test_df = test_df.drop(columns=['month'])
    
    return train_df, val_df, test_df


def r_to_dbz(r):
    """Convert mm/h to dBZ in place, skipping zeros."""
    if isinstance(r, np.ndarray):
        mask = r != 0  # Mask to exclude zero elements
        r[mask] = 10 * np.log10(200*r[mask]**(8/5)+1) 

    elif isinstance(r, torch.Tensor):
        mask = r != 0
        r[mask] = 10 * torch.log10(200*r[mask]**(8/5)+1) 
    else:
        raise TypeError('Array must be a numpy array or torch Tensor.')
    return r

def dbz_to_r(dbz):
    """Convert dBZ to mm/h in place, skipping zeros."""
    if isinstance(dbz, np.ndarray):
        mask = dbz != 0  # Mask to exclude zero elements
        dbz[mask] = ((10**(dbz[mask]/10)-1)/200)**(5/8)
    elif isinstance(dbz, torch.Tensor):
        mask = dbz != 0
        dbz[mask] = ((10**(dbz[mask]/10)-1)/200)**(5/8)
    else:
        raise TypeError('Array must be a numpy array or torch Tensor.')
    return dbz

def data_prep(x, norm_method='minmax', convert_to_dbz=False, undo=False):
    """Perform in-place data preprocessing with optimized speed, skipping zeros."""
    assert norm_method in ('minmax', 'minmax_tanh')

    MIN, MAX = 0, 100
    if convert_to_dbz:
        MAX = 55

    if not undo:
        if convert_to_dbz:
            r_to_dbz(x)  # Convert in place, skipping zeros

        np.clip(x, MIN, MAX, out=x)

        if norm_method == 'minmax_tanh':
            x -= (MIN + MAX / 2)
            x /= (MAX / 2 - MIN)
        else:
            x -= MIN
            x /= (MAX - MIN)
    else:
        if norm_method == 'minmax_tanh':
            x *= (MAX / 2 - MIN)
            x += (MIN + MAX / 2)
        else:
            x *= (MAX - MIN)
            x += MIN

        if convert_to_dbz:
            np.copyto(x, dbz_to_r(x))  # Modify x in place, skipping zeros

    return x

def random_crop(x, crop_h, crop_w):
    """
    Randomly crop a region of size (crop_h, crop_w) from a 4D NumPy array 
    with shape (channels, time, height, width).
    """
    channels, time, height, width = x.shape

    assert crop_h <= height, f"crop_h ({crop_h}) must be <= image height ({height})"
    assert crop_w <= width, f"crop_w ({crop_w}) must be <= image width ({width})"

    top = np.random.randint(0, height - crop_h + 1)
    left = np.random.randint(0, width - crop_w + 1)

    cropped_x = x[:, :, top : top + crop_h, left : left + crop_w]
    return cropped_x


class RadarDataset(Dataset):
    def __init__(
        self,
        df: pd.DataFrame,
        ds: dict[str, xr.Dataset] | xr.Dataset,
        context_len: int = 4,
        forecast_len: int = 18,
        train_encoder: bool = False
    ):
        self.df = df
        self.ds = ds
        self.context_len = context_len
        self.forecast_len = forecast_len
        self.train_encoder = train_encoder

    def __len__(self):
        return len(self.df)
    
    def process_images(self, selected_images: np.ndarray) -> np.ndarray:
        """Image processing"""
        # Scale to mm/h - vectorized operation
        x = selected_images.astype(np.float32) * 0.12  # (Time, H, W)

        data_prep(x, convert_to_dbz=True)

        x = np.expand_dims(x, axis=0)

        x = random_crop(x, 128, 128)
        
        return x

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        first_index = row['first_index']
        last_index = row['last_index']

        if isinstance(self.ds, dict):
            group, variable = row['image_data_path'].split("/")
            image_data = self.ds[group][variable]
        else:
            image_data = self.ds[row['image_data_path']]

        # Ajust sample length
        if self.context_len+self.forecast_len < last_index-first_index:
            last_index = first_index + self.context_len+self.forecast_len-1

        selected_images = image_data.isel(time=slice(first_index, last_index + 1)).to_numpy()

        # Process the selected images.
        processed_images = self.process_images(selected_images)

        if self.train_encoder:
            return (processed_images, processed_images)

        # Split processed images into input sequence and future targets.
        context_images = processed_images[:, :self.context_len, :, :]
        future_images = processed_images[:, self.context_len:, :, :]
        
        timesteps = np.arange(-self.context_len+1,1, dtype=np.float32)

        return ([[context_images, timesteps]], future_images)
    
class RadarDataModule(pl.LightningDataModule):
    def __init__(
        self,
        df_path: str,
        ds_path: str,
        ds_groups: Optional[List] = None, 
        context_len: int = 4,
        forecast_len: int = 18,
        train_encoder: bool = False,
        test_months: List[int] = [],
        val_months: List[int] = [],
        batch_size: int = 4,
        num_workers: int = 8,
    ):
        super().__init__()
        self.df_path = df_path
        self.ds_path = ds_path
        self.ds_groups = ds_groups
        self.context_len = context_len
        self.forecast_len = forecast_len
        self.train_encoder = train_encoder
        self.test_months = test_months
        self.val_months = val_months
        self.batch_size = batch_size
        self.num_workers = num_workers

    def setup(self, stage: Optional[str] = None):
        
        df = pd.read_parquet(self.df_path)

        if self.ds_groups:
            ds = {}
            for group in self.ds_groups:
                ds[group] = xr.open_dataset(self.ds_path, group=group)
        else:
            ds = xr.open_dataset(self.ds_path)

        train_df, val_df, test_df = split_df_by_months(
                                        df, 
                                        self.val_months, 
                                        self.test_months
                                    )
    
        self.train_dataset = RadarDataset(
            df=train_df,
            ds=ds,
            context_len=self.context_len,
            forecast_len=self.forecast_len,
            train_encoder=self.train_encoder
        )
        self.val_dataset = RadarDataset(
            df=val_df,
            ds=ds,
            context_len=self.context_len,
            forecast_len=self.forecast_len,
            train_encoder=self.train_encoder
            )

        # === Adding the bin column and computing weights for train data ===
        # Define the bins (rain rates on a log scale)
        bins = np.exp(np.linspace(np.log(0.2), np.log(15), 6))
        bin_edges = np.concatenate(([0], bins, [np.inf]))

        # Use pd.cut to assign each sample to a bin based on max_intensity.
        # The resulting bin labels are integers 0 to num_bins-1.
        df_bins = pd.cut(train_df['max_intensity'], bins=bin_edges, labels=False, include_lowest=True)

        # Compute counts per bin
        bin_counts = df_bins.value_counts().sort_index()
        # Compute weight for each sample as inverse frequency of its bin.
        weights = torch.tensor(df_bins.map(lambda b: 1.0 / bin_counts[b]).values, dtype=torch.float)
        self.sampler = WeightedRandomSampler(weights, num_samples=100000, replacement=True)
        
    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            sampler=self.sampler,
            pin_memory=True,
            persistent_workers=self.num_workers > 0
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

# Example usage:
if __name__ == '__main__':
    dataset_path = "/vol/csedu-nobackup/project/mrobben/nowcasting/dataset/dataset.nc" 
    parque_path = "/vol/csedu-nobackup/project/mrobben/nowcasting/dataset/samples.parquet"

    df = pd.read_parquet(parque_path)

    groups = ['Crop1','Crop2','Crop3','Crop4','Crop5','Crop6','Crop7','Crop8']
    ds = {}
    for group in groups:
        ds[group] = xr.open_dataset(dataset_path, group=group)

    dataset = RadarDataset(df=df, ds=ds, context_len=4, forecast_len=18)
    dataloader = DataLoader(dataset, batch_size=1, num_workers=1, shuffle=True)

    # val = [
    #     # "2020-11", "2020-12", "2021-01", "2021-02", "2021-03"
    #     "202005", "202011", "202105", "202111", "202205", "202211"
    #     ]
    # test = [
    #     # "2021-01", "2021-02", "2021-03", "2021-04", "2021-05", "2021-06", "2021-07", "2021-08", "2021-09", "2021-10", "2021-11", "2021-12",
    #     # "2022-01", "2022-02", "2022-03", "2022-04", "2022-05", "2022-06", "2022-07", "2022-08", "2022-09", "2022-10", "2022-11", "2022-12", 
    #     "202301", "202302", "202303", "202304", "202305", "202306", "202307", "202308", "202309", "202310", "202311", "202312", "202401"
    #     ]
    
    # # val = ["2020-11", "2020-12"]
    # # test = []

    # # Parameters
    # context_frames = 4
    # forecast_steps = 1
    
    # datamodule = RadarDataModule(
    #     df_path="dataset/samples.parquet",
    #     ds_path="dataset/dataset.nc", 
    #     ds_groups=['Crop1','Crop2','Crop3','Crop4','Crop5','Crop6','Crop7','Crop8'],
    #     context_len=context_frames,
    #     forecast_len=forecast_steps,
    #     batch_size=4, # 7 with discriminators
    #     num_workers=4,
    #     val_months=val,
    #     test_months=test
    # )
    # datamodule.setup()

    # dataloader = datamodule.train_dataloader()

    st = time.time()
    # Iterate over the DataLoader.
    for batch_idx, (images, future_images) in enumerate(dataloader):
        pass
        # print(f"Batch {batch_idx}: images shape: {images.shape}, timestamps shape: {future_images.shape}, type: {type(images)}")
        # Pass the batch to your model, or perform additional processing.
        # For demonstration, we'll break after one batch.
        if batch_idx == 1000:
            break

    et = time.time()    
    print(et-st)

    for group in groups:
        ds[group].close()
