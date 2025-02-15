import argparse
import yaml
import logging
import os
import random
import torch
from omegaconf import DictConfig, OmegaConf

# Set up logging
logger = logging.getLogger(__name__)


def load_config(config_file: str) -> DictConfig:
    """
    Load the configuration data from the YAML file.

    Args:
        config_file (str): Path to the configuration YAML file.

    Returns:
        DictConfig: The combined configuration dictionary.
    """
    config = OmegaConf.load(config_file)
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_name", type=str, default=config.defaults[0].dataset, help="Name of the dataset")
    args = parser.parse_args()

    dataset_name = args.dataset_name
    if dataset_name not in config.dataset:
        raise ValueError(f"Dataset '{dataset_name}' not found in configuration file.")
    
    cfg = OmegaConf.merge(config.common, config.dataset[dataset_name])
    return prepare_config(cfg)

def prepare_config(cfg: DictConfig) -> DictConfig:
    """
    Prepare the configuration.

    Args:
        cfg (DictConfig): Configuration dictionary.

    Returns:
        DictConfig: The updated configuration dictionary.
    """
    logger.info("Preparing configuration...")
    
    check_dir(cfg.run_dir)
    check_seed(cfg.seed)
    cfg.device = check_device()

    # Log hyper-parameters
    logger.info("*******  Hyper-parameters  ********")
    logger.info(OmegaConf.to_yaml(cfg))
    logger.info("***********************************")

    logger.info("Configuration prepared successfully")
    return cfg

def check_dir(dir_name: str) -> None:
    """
    Check if a directory exists, and if not, create it.

    Args:
        dir_name (str): The directory path to check or create.
    """
    os.makedirs(dir_name, exist_ok=True)
    logger.info(f"Directory verified: {dir_name}")

def check_seed(seed: int) -> None:
    """
    Set the random seed for reproducibility.

    Args:
        seed (int): The random seed value.
    """
    random.seed(seed)
    torch.manual_seed(seed)
    logger.info(f"Random seed set to: {seed}")

def check_device() -> str:
    """
    Check if CUDA is available and return the appropriate device.

    Returns:
        str: The device to use ('cuda' or 'cpu').
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Device set to: {device}")
    return device