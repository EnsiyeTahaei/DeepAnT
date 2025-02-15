import logging
import os
from datetime import datetime

from utils.utils import load_config
from utils.data_utils import load_data
from deepant.trainer import DeepAnT


# Set up logging
LOG_DIR = "logs"
os.makedirs(LOG_DIR, exist_ok=True)

log_filename = os.path.join(LOG_DIR, f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_filename),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def main():
    """
    Main entry point for running the DeepAnT anomaly detection pipeline.

    This script:
      1. Loads a configuration (config.yaml).
      2. Loads and preprocesses the dataset.
      3. Initializes and trains the DeepAnT model.
      4. Uses the trained model to detect anomalies.

    Usage:
        python main.py

    Note:
        You may customize arguments such as `dataset_name` and `device` inside `config.yaml`.
    """
    config = load_config("config.yaml")
    logger.info(f"Starting DeepAnT with dataset: {config['dataset_name']}")

    # Load the dataset
    logger.info("Loading data...")
    train_dataset, val_dataset, test_dataset, feature_dim = load_data(
        config["dataset_name"], 
        config["window_size"], 
        config["device"]
    )
    logger.info("Data loaded successfully")

    model = DeepAnT(config, train_dataset, val_dataset, test_dataset, feature_dim)
    model.train()
    model.detect_anomaly()


if __name__ == "__main__":
    main()
