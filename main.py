import logging
import os
from datetime import datetime
from utils.utils import load_config
from utils.data_utils import load_data
from deepant.trainer import DeepAnT


# Set up logging
log_dir = "logs"
if not os.path.exists(log_dir):
    os.makedirs(log_dir)

log_filename = os.path.join(log_dir, f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")

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
    Main function to run the DeepAnT anomaly detection model.
    """
    config = load_config("config.yaml")
    logger.info(f"Starting DeepAnT with dataset: {config['dataset_name']}")

    # Loading dataset
    logger.info("Loading data...")
    train_dataset, val_dataset, test_dataset, feature_dim = load_data(
        config["dataset_name"], config["window_size"], config["device"]
    )
    logger.info("Data loaded successfully")

    # Initialize and train model
    model = DeepAnT(config, train_dataset, val_dataset, test_dataset, feature_dim)
    model.train()
    model.detect_anomaly()
    logger.info("Anomaly detection procedure completed")

if __name__ == "__main__":
    main()
