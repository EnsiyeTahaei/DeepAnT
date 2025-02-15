import os
import logging
import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import MinMaxScaler

# Set up logging
logger = logging.getLogger(__name__)

class DataModule(torch.utils.data.Dataset):
    def __init__(self, data_x, data_y, device):
        """
        PyTorch Dataset for time series data.

        Args:
            data_x (np.ndarray): The input data.
            data_y (np.ndarray): The target data.
            device (str): The device to use ('cuda' or 'cpu').
        """
        self.data_x = data_x
        self.data_y = data_y
        self.device = device

    def __len__(self):
        return len(self.data_x)

    def __getitem__(self, idx):
        return (
            torch.tensor(self.data_x[idx], device=self.device, dtype=torch.float32).transpose(0, 1),
            torch.tensor(self.data_y[idx], device=self.device, dtype=torch.float32).squeeze(0),
        )

def load_data(dataset_name, window_size, device, val_rate=0.1, test_rate=0.1):
    """
    Load and preprocess the dataset.

    Args:
        dataset_name (str): Name of the dataset.
        window_size (int): The size of the sliding window.
        device (str): The device to use ('cuda' or 'cpu').
        val_rate (float): The validation set split rate.
        test_rate (float): The test set split rate.

    Returns:
        tuple: Train, validation, and test datasets, and the feature dimension of the dataset.
    """
    logger.info(f"Loading dataset: {dataset_name}")
    path = os.path.join("./data", dataset_name)

    if dataset_name == "NAB":
        file_name = "TravelTime_451.csv"
        data = pd.read_csv(os.path.join(path, file_name), index_col="timestamp", parse_dates=["timestamp"])

    elif dataset_name == "AirQuality":
        file_name = "AirQualityUCI.csv"
        data = pd.read_csv(os.path.join(path, file_name), sep=";", decimal=".", na_values=-200)
        data["timestamp"] = pd.to_datetime(data["Date"] + " " + data["Time"], format="%d/%m/%Y %H.%M.%S")
        data.drop(columns=["Date", "Time"], inplace=True)
        data.set_index("timestamp", inplace=True)
        data.dropna(axis=1, how="all", inplace=True)
        data.ffill(inplace=True)
        data.dropna(axis=0, how="any", inplace=True)
        data = data.select_dtypes(include=[np.number])
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

def split_data(data, window_size):
    """
    Split the data using sliding windows.

    Args:
        data (np.ndarray): The input data.
        window_size (int): The size of the sliding window.

    Returns:
        tuple: Arrays of input and target data.
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