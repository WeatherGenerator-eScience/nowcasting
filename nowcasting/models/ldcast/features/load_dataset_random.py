from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import torch
import pytorch_lightning as pl

import xarray as xr
import numpy as np
import pandas as pd

from typing import Optional, List, Tuple

import time

class RandomRadarDataset(Dataset):
    def __init__(self, num_samples: int, context_len: int, forecast_len: int, channels: int, height: int, width: int, train_encoder: bool = False):
        """
        Args:
            num_samples (int): Total number of synthetic samples to generate.
            context_len (int): Number of context frames.
            forecast_len (int): Number of forecast frames.
            channels (int): Number of channels per image.
            height (int): Image height.
            width (int): Image width.
        """
        self.num_samples = num_samples
        self.context_len = context_len
        self.forecast_len = forecast_len
        self.channels = channels
        self.height = height
        self.width = width
        self.train_encoder = train_encoder

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        # Generate a random tensor of shape (Seq Length, Channels, Height, Width)
        if self.train_encoder:
            # Encoder case: only context frames
            context = torch.randn(self.channels[0], self.context_len, self.height, self.width)
            return (context, context)

        timesteps = np.arange(-self.context_len+1,1, dtype=np.float32)

        context0 = torch.randn(self.channels[0], self.context_len, self.height, self.width)
        context1 = torch.randn(self.channels[1], self.context_len, self.height, self.width)
        context2 = torch.randn(self.channels[2], self.context_len, self.height//2, self.width//2)
        context3 = torch.randn(self.channels[3], self.context_len, self.height//2, self.width//2)
        context4 = torch.randn(self.channels[4], self.context_len, self.height//2, self.width//2)
        future = torch.randn(self.channels[0], self.forecast_len, self.height, self.width)
        
        return ([[context0, timesteps],
                 [context1, timesteps],
                 [context2, timesteps],
                 [context3, timesteps],
                 [context4, timesteps],
                 ], future) 

class RandomRadarDataModule(pl.LightningDataModule):
    def __init__(
        self,
        num_samples_train: int,
        num_samples_val: int,
        context_len: int,
        forecast_len: int,
        channels: list[int],
        height: int,
        width: int,
        train_encoder: bool = False,
        batch_size: int = 4,
        num_workers: int = 4,
    ):
        """
        DataModule to generate synthetic data for benchmarking.
        
        Args:
            num_samples (int): Number of samples in the dataset.
            context_len (int): Number of context frames.
            forecast_len (int): Number of forecast frames.
            channels (int): Number of channels.
            height (int): Image height.
            width (int): Image width.
            batch_size (int): Batch size.
            num_workers (int): Number of workers for data loading.
        """
        super().__init__()
        self.num_samples_train = num_samples_train
        self.num_samples_val = num_samples_val
        self.context_len = context_len
        self.forecast_len = forecast_len
        self.channels = channels
        self.height = height
        self.width = width
        self.train_encoder = train_encoder
        self.batch_size = batch_size
        self.num_workers = num_workers

    def setup(self, stage: str = None):
        # Initialize the random dataset
        self.dataset_train = RandomRadarDataset(
            num_samples=self.num_samples_train,
            context_len=self.context_len,
            forecast_len=self.forecast_len,
            channels=self.channels,
            height=self.height,
            width=self.width,
            train_encoder=self.train_encoder,
        )

        self.dataset_val = RandomRadarDataset(
            num_samples=self.num_samples_val,
            context_len=self.context_len,
            forecast_len=self.forecast_len,
            channels=self.channels,
            height=self.height,
            width=self.width,
            train_encoder=self.train_encoder,
        )

    def train_dataloader(self):
        return DataLoader(
            self.dataset_train,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.dataset_val,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=True,
        )