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
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_name", type=str, default="nab", help="Name of the dataset")
    args = parser.parse_args()

    with open(config_file, 'r') as file:
        config = yaml.safe_load(file)

    # Select the appropriate dataset configuration
    if args.dataset_name in config['dataset']:
        dataset_cfg = config['dataset'][args.dataset_name]
    else:
        raise ValueError(f"Dataset {args.dataset_name} not found in configuration file.")
    
    common_cfg = config['common']
    combined_cfg = {**common_cfg, **dataset_cfg}
    return prepare_config(OmegaConf.create(combined_cfg))

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
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)
    logger.info(f"Directory checked/created: {dir_name}")

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