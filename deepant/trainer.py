import os
import json
import numpy as np
import logging
import torch
import matplotlib.pyplot as plt
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from deepant.model import DeepAntPredictor

# Set up logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class AnomalyDetector(pl.LightningModule):
    def __init__(self, model, lr):
        """
        Anomaly Detector based on DeepAnt.

        Args:
            model (nn.Module): The DeepAnt predictor model.
            lr (float): Learning rate for the optimizer.
        """
        super(AnomalyDetector, self).__init__()
        self.model = model
        self.criterion = torch.nn.L1Loss()
        self.lr = lr

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_pred = self(x)
        loss = self.criterion(y_pred, y)
        self.log("train_loss", loss.item(), on_epoch=True)
        logger.info(f"Epoch {self.current_epoch} - Training step {batch_idx} - Loss: {loss.item()}")
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_pred = self(x)
        loss = self.criterion(y_pred, y)
        self.log("val_loss", loss.item(), on_epoch=True)
        logger.info(f"Epoch {self.current_epoch} - Validation step {batch_idx} - Loss: {loss.item()}")
        return loss

    def predict_step(self, batch, batch_idx):
        x, y = batch
        y_pred = self(x)
        logger.info(f"Prediction step {batch_idx} - Predictions: {y_pred}")
        return y_pred

    def configure_optimizers(self):
        logger.info(f"Configuring optimizer with learning rate: {self.lr}")
        return torch.optim.Adam(self.parameters(), lr=self.lr)


class DeepAnT:
    def __init__(self, config, train_dataset, val_dataset, test_dataset, feature_dim):
        """
        DeepAnT class.

        Args:
            config (DictConfig): Configuration dictionary.
            train_dataset (Dataset): Training dataset.
            val_dataset (Dataset): Validation dataset.
            test_dataset (Dataset): Test dataset.
            feature_dim (int): Number of channels in the input data.
        """
        self.config = config
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.test_dataset = test_dataset
        self.feature_dim = feature_dim
        self.device = config["device"]

        # Initialize the model
        self.deepant_predictor = DeepAntPredictor(
            feature_dim=feature_dim,
            window_size=config["window_size"],
            hidden_size=config["hidden_size"]
        ).to(self.device)

        self.anomaly_detector = AnomalyDetector(
            model=self.deepant_predictor,
            lr=float(config["lr"])
        ).to(self.device)

        # Initial trainer configuration
        self.initial_trainer = pl.Trainer(
            max_epochs=config["max_initial_steps"],
            callbacks=[
                pl.callbacks.ModelCheckpoint(
                    dirpath=config["run_dir"],
                    filename="initial_model",
                    monitor="epoch",
                    save_top_k=1,
                    mode="max"
                )
            ],
            default_root_dir=config["run_dir"],
            accelerator=self.device,
            devices=1 if self.device == "cuda" else "auto",
        )

        # Main trainer configuration
        self.trainer = pl.Trainer(
            max_epochs=config["max_steps"],
            callbacks=[
                pl.callbacks.ModelCheckpoint(
                    dirpath=config["run_dir"],
                    filename="best_model",
                    monitor="val_loss",
                    save_top_k=1,
                    mode="min"
                ),
                pl.callbacks.EarlyStopping(
                    monitor="val_loss",
                    patience=config["patience"],
                    mode="min"
                )
            ],
            default_root_dir=config["run_dir"],
            accelerator=self.device,
            devices=1 if self.device == "cuda" else "auto",
        )

        logger.info("DeepAnT model initialized.")

    def train(self):
        """
        Train the DeepAnT model.
        """
        train_loader = DataLoader(self.train_dataset, batch_size=self.config["batch_size"], shuffle=True)
        val_loader = DataLoader(self.val_dataset, batch_size=self.config["batch_size"], shuffle=False)

        # Initial training phase
        logger.info("Starting initial training phase...")
        self.initial_trainer.fit(self.anomaly_detector, train_loader)

        # Load the initial model checkpoint for main training phase
        initial_checkpoint_path = os.path.join(self.config["run_dir"], "initial_model.ckpt")
        self.anomaly_detector = AnomalyDetector.load_from_checkpoint(
            checkpoint_path=initial_checkpoint_path,
            model=self.deepant_predictor,
            lr=float(self.config["lr"])
        )

        # Main training phase
        logger.info("Starting main training phase...")
        self.trainer.fit(self.anomaly_detector, train_loader, val_loader)
        logger.info("Training completed.")

    def detect_anomaly(self):
        """
        Detect anomalies using the DeepAnT trained model.
        """
        logger.info("Starting anomaly detection process...")
        test_loader = DataLoader(self.test_dataset, batch_size=self.test_dataset.data_x.shape[0], shuffle=False)
        best_model = AnomalyDetector.load_from_checkpoint(
            checkpoint_path=os.path.join(self.config["run_dir"], "best_model.ckpt"),
            model=self.deepant_predictor,
            lr=float(self.config["lr"])
        )
        output = self.trainer.predict(best_model, test_loader)

        ground_truth = test_loader.dataset.data_y.squeeze()
        predictions = output[0].numpy().squeeze()

        anomaly_scores = [np.linalg.norm(x - y) for x, y in zip(predictions, ground_truth)]
        threshold = self.calculate_threshold(anomaly_scores)
        logger.info(f"Calculated threshold for anomaly detection: {threshold}")

        # Anomalies identification
        anomalies, anomalies_indices = self.identify_anomalies(anomaly_scores, threshold)
        with open(os.path.join(self.config["run_dir"], "anomalies_indices.json"), "w") as json_file:
            json.dump(anomalies_indices, json_file)
        logger.info(f"Anomalies detected at indices: {anomalies}")

        self.visualize_results(self.test_dataset, ground_truth, predictions, anomalies, anomaly_scores, threshold)
        logger.info("Anomaly detection process completed.")

    def calculate_threshold(self, anomaly_scores, std_rate=2):
        """
        Calculate a dynamic threshold for anomaly detection.

        Args:
            anomaly_scores (list): List of anomaly scores.
            std_rate (int, optional): Standard deviation multiplier for threshold (Defaults to 2).

        Returns:
            float: Calculated threshold value.
        """
        mean_scores = np.mean(anomaly_scores)
        std_scores = np.std(anomaly_scores)
        return mean_scores + std_rate * std_scores

    def identify_anomalies(self, anomaly_scores, threshold):
        """
        Identify anomalies based on the calculated threshold.

        Args:
            anomaly_scores (list): List of anomaly scores.
            threshold (float): Calculated threshold value for anomaly detection.

        Returns:
            tuple: List of anomaly indices and a dictionary of anomalies.
        """
        anomalies_dict = {}
        if self.feature_dim == 1:
            anomalies = [i for i, score in enumerate(anomaly_scores) if score > threshold]
            anomalies_dict["Feature_1"] = anomalies
        print("Anomalies Detected: ", anomalies_dict)
        return anomalies, anomalies_dict

    def visualize_results(self, dataset, target_seq, pred_seq, anomalies, anomaly_scores, threshold):
        """
        Visualize the results (predictions and anomaly detections)

        Args:
            dataset (DataModule): The test dataset.
            target_seq (np.ndarray): The target sequence.
            pred_seq (np.ndarray): The predicted sequence.
            anomalies (list): List of anomaly indices.
            anomaly_scores (list): List of anomaly scores.
            threshold (float): The calculated threshold value.
        """
        # Reconstruct the input sequence from the sliding windows
        original_data = dataset.data_x[:, 0, 0]
        time_steps = range(len(original_data) + len(pred_seq))

        # Plotting the sequences
        plt.figure(figsize=(14, 7))
        plt.plot(time_steps[:len(original_data)], original_data, label="Input Sequence", color="blue", linewidth=1.5)
        plt.plot(time_steps[len(original_data):], target_seq, label="Target Sequence", color="green", linestyle='--', linewidth=1.5)
        plt.plot(time_steps[len(original_data):], pred_seq, label="Predicted Sequence", color="orange", linestyle='-.', linewidth=1.5)
        plt.title(f"Sinewave Prediction on the {self.config['dataset_name']} dataset", fontsize=16)
        plt.xlabel("Time", fontsize=14)
        plt.ylabel("Value", fontsize=14)
        plt.legend(loc="upper right", fontsize=12)
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.tight_layout()
        plt_name = "predictions"
        plt.savefig(self.config["run_dir"] + "/" + plt_name + ".png")
        plt.close()

        # Plotting the anomaly scores
        plt.figure(figsize=(12, 6))
        plt.plot(time_steps[len(original_data):], anomaly_scores, label="Anomaly Scores", color='blue', linestyle='-', linewidth=1)
        plt.scatter([time_steps[len(original_data) + i] for i in anomalies], [anomaly_scores[i] for i in anomalies], color='red', label='Anomaly Points', zorder=5)
        plt.axhline(y=threshold, color='r', linestyle='--', linewidth=1, label="Threshold")
        plt.title(f"Anomaly Scores and Threshold on the {self.config['dataset_name']} dataset", fontsize=16)
        plt.xlabel("Time", fontsize=14)
        plt.ylabel("Score", fontsize=14)
        plt.legend(loc="upper right", fontsize=12)
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.tight_layout()
        plt_name = "anomalies"
        plt.savefig(self.config["run_dir"] + "/" + plt_name + ".png")
        plt.close()
