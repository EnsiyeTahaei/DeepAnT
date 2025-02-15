"""
Data Utilities for Time-Series Datasets

"""

import os
import logging
import numpy as np
import pandas as pd
import torch
from typing import Tuple
from sklearn.preprocessing import MinMaxScaler

# Set up logging
logger = logging.getLogger(__name__)


class DataModule(torch.utils.data.Dataset):
    def __init__(self, data_x: np.ndarray, data_y: np.ndarray, device: str) -> None:
        """
        PyTorch Dataset for time series data.
        
        Each sample is a tuple:
            (input_sequence, target_value)
        where:
            - input_sequence: shape (num_features, window_size)
            - target_value: shape (num_features,)

        Args:
            data_x (np.ndarray): Array of input sequences, 
                                 shape (num_samples, window_size, num_features).
            data_y (np.ndarray): Array of target values (one step ahead), 
                                 shape (num_samples, 1, num_features).
            device (str): Compute device to use ('cuda' or 'cpu').
        """
        self.data_x = data_x
        self.data_y = data_y
        self.device = device

    def __len__(self) -> int:
        """
        Returns:
            int: The total number of samples.
        """
        return len(self.data_x)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Retrieves a single time-series sample and corresponding target value.
        
        Args:
            idx (int): Index of the sample.
        
        Returns:
            Tuple[torch.Tensor, torch.Tensor]:
                - input_sequence (torch.Tensor): Shape (num_features, window_size)
                - target_value (torch.Tensor):   Shape (num_features,)
        """
        return (
            torch.tensor(self.data_x[idx], device=self.device, dtype=torch.float32).transpose(0, 1),
            torch.tensor(self.data_y[idx], device=self.device, dtype=torch.float32).squeeze(0),
        )


def load_data(
        dataset_name: str, 
        window_size: int, 
        device: str, 
        val_rate: float = 0.1, 
        test_rate: float = 0.1
) -> Tuple[DataModule, DataModule, DataModule, int]:
    """
    Load and preprocess a dataset.

    Args:
        dataset_name (str): Name of the dataset (e.g., 'NAB', 'AirQuality').
        window_size (int): Number of time steps per input window.
        device (str): Compute device to use ('cuda' or 'cpu').
        val_rate (float, optional): Fraction of data used for validation
                                    (Defaults to 0.1).
        test_rate (float, optional): Fraction of data used for testing
                                     (Defaults to 0.1).

    Returns:
        Tuple[DataModule, DataModule, DataModule, int]: 
            - train_dataset (DataModule): Training dataset.
            - val_dataset (DataModule): Validation dataset.
            - test_dataset (DataModule): Testing dataset.
            - num_features (int): Number of features in the dataset.
    """
    logger.info(f"Loading dataset: {dataset_name}")
    path = os.path.join("./data", dataset_name)

    if dataset_name == "NAB":
        file_name = "TravelTime_451.csv"
        data = pd.read_csv(
            os.path.join(path, file_name),
            index_col="timestamp",
            parse_dates=["timestamp"]
        )

    elif dataset_name == "AirQuality":
        file_name = "AirQualityUCI.csv"
        data = pd.read_csv(
            os.path.join(path, file_name),
            sep=";",
            decimal=".",
            na_values=-200      # As per dataset docs: -200 indicates missing data
        )
        data["timestamp"] = pd.to_datetime(
            data["Date"] + " " + data["Time"], format="%d/%m/%Y %H.%M.%S"
        )
        data.drop(columns=["Date", "Time"], inplace=True)
        data.set_index("timestamp", inplace=True)
        data.dropna(axis=1, how="all", inplace=True)        # Remove columns fully NaN
        data.ffill(inplace=True)                            # Forward-fill missing values 
        data.dropna(axis=0, how="any", inplace=True)        # Drop rows that still have NaNs
        data = data.select_dtypes(include=[np.number])      # Keep only numeric columns (sensor data)
    else:
        raise ValueError(f"Unknown dataset name: {dataset_name}")

    logger.info(f"Dataset shape after loading: {data.shape}")

    sc = MinMaxScaler()
    data_scaled = sc.fit_transform(data)
    data_x, data_y = split_data(data_scaled, window_size)

    train_slice = slice(None, int((1 - val_rate - test_rate) * len(data_x)))
    val_slice = slice(int((1 - val_rate - test_rate) * len(data_x)), int((1 - test_rate) * len(data_x)))
    test_slice = slice(int((1 - test_rate) * len(data_x)), None)

    train_dataset = DataModule(data_x[train_slice], data_y[train_slice], device)
    val_dataset = DataModule(data_x[val_slice], data_y[val_slice], device)
    test_dataset = DataModule(data_x[test_slice], data_y[test_slice], device)

    logger.info(f"Train dataset shape: {train_dataset.data_x.shape}")
    logger.info(f"Validation dataset shape: {val_dataset.data_x.shape}")
    logger.info(f"Test dataset shape: {test_dataset.data_x.shape}")

    return train_dataset, val_dataset, test_dataset, data_x.shape[-1]

def split_data(
        data: np.ndarray, 
        window_size: int
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Split the data using sliding windows, with the next step as the target.

    Args:
        data (np.ndarray): Scaled data of shape (num_samples, num_features).
        window_size (int): Number of time steps in each window.

    Returns:
        Tuple[np.ndarray, np.ndarray]: 
            - data_x (np.ndarray): Sliding windows of input data,  
                                   with shape (num_windows, window_size, num_features).
            - data_y (np.ndarray): Next-step targets with shape (num_windows, 1, num_features).
    """
    data_x, data_y = [], []
    for i in range(window_size, data.shape[0]):
        if (i + 1) >= data.shape[0]:
            break
        window = data[i - window_size:i]
        target = data[i:i + 1] 
        data_x.append(window)
        data_y.append(target)

    return np.array(data_x), np.array(data_y)