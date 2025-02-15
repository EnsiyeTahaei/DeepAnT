import os
import json
import numpy as np
import logging
import torch
import matplotlib.pyplot as plt
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from deepant.model import DeepAntPredictor
from typing import Dict, List

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
        return torch.optim.Adam(self.parameters(), lr=self.lr, weight_decay=1e-4)


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
        if ground_truth.ndim == 1:
            ground_truth = ground_truth.reshape(-1, 1)
        if predictions.ndim == 1:
            predictions = predictions.reshape(-1, 1)

        anomaly_scores = np.abs(predictions - ground_truth)
        if anomaly_scores.ndim == 1:
            anomaly_scores = anomaly_scores.reshape(-1, 1)

        thresholds = self.calculate_thresholds(anomaly_scores)
        logger.info(f"Calculated thresholds are: {thresholds}")

        anomalies_indices = self.identify_anomalies(anomaly_scores, thresholds)
        with open(os.path.join(self.config["run_dir"], "anomalies_indices.json"), "w") as json_file:
            json.dump(anomalies_indices, json_file)
        logger.info(f"Anomalies detected: {anomalies_indices}")

        self.visualize_results(ground_truth, predictions, anomalies_indices, thresholds)
        logger.info("Anomaly detection process completed.")

    def calculate_thresholds(self, anomaly_scores, std_rate=2):
        """
        Calculate a dynamic threshold for anomaly detection.

        Args:
            anomaly_scores (list): List of anomaly scores.
            std_rate (int, optional): Standard deviation multiplier for threshold (Defaults to 2).

        Returns:
            float: Calculated threshold value.
        """
        thresholds = []
        for feature_idx in range(self.feature_dim):
            feature_scores = anomaly_scores[:, feature_idx]
            mean_scores = np.mean(feature_scores)
            std_scores = np.std(feature_scores)
            thresholds.append(mean_scores + std_rate * std_scores)
        return thresholds

    def identify_anomalies(self, anomaly_scores, thresholds):
        """
        Identify anomalies based on the calculated threshold.

        Args:
            anomaly_scores (list): List of anomaly scores.
            threshold (float): Calculated threshold value for anomaly detection.

        Returns:
            tuple: List of anomaly indices and a dictionary of anomalies.
        """
        anomalies_dict = {}
        for f_idx in range(self.feature_dim):
            feature_scores = anomaly_scores[:, f_idx]
            anomalies_dict[f"Feature_{f_idx + 1}"] = [i for i, score in enumerate(feature_scores) if score > thresholds[f_idx]]
        return anomalies_dict
    
    def reconstruct_original_sequence(self):
        val_windows = self.val_dataset.data_x
        val_reconstructed_list = []
        for i in range(len(val_windows)):
            window = val_windows[i]
            if i == 0:
                val_reconstructed_list.append(window)
            else:
                val_reconstructed_list.append(window[-1:])

        val_reconstructed = np.concatenate(val_reconstructed_list, axis=0)

        test_windows = self.test_dataset.data_x
        test_reconstructed_list = []
        for i in range(len(test_windows)):
            window = test_windows[i]
            test_reconstructed_list.append(window[-1:])

        test_reconstructed = np.concatenate(test_reconstructed_list, axis=0)

        boundary_idx = len(val_reconstructed)
        full_seq = np.concatenate([val_reconstructed, test_reconstructed], axis=0)

        return full_seq, boundary_idx

    def visualize_results(self, target_seq, pred_seq, anomalies, thresholds):
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
        if len(thresholds) == 1 and self.feature_dim > 1:
            logger.warning("A single threshold was provided for multi-feature data.")

        original_data, boundary_idx = self.reconstruct_original_sequence()
        time_steps = range(original_data.shape[0])

        fig, axs = plt.subplots(self.feature_dim, 2, figsize=(28, 7 * self.feature_dim), sharex=False, sharey=False)
        if self.feature_dim == 1:
            axs = axs.reshape(1, 2)
        axs[0, 0].annotate("Target vs. Prediction", xy=(0.5, 1.15), xycoords='axes fraction', 
            ha='center', va='bottom', fontsize=14, weight='bold', annotation_clip=False)
        axs[0, 1].annotate("Detected Anomalies", xy=(0.5, 1.15), xycoords='axes fraction', 
            ha='center', va='bottom', fontsize=14, weight='bold', annotation_clip=False)

        handles_left, labels_left = [], []
        handles_right, labels_right = [], []
        
        for f_idx in range(self.feature_dim):
            input_label = "Validation Sequence" if f_idx == 0 else "_nolegend_"
            target_label = "Target Sequence" if f_idx == 0 else "_nolegend_"
            pred_label = "Predicted Sequence" if f_idx == 0 else "_nolegend_"

            line1 = axs[f_idx, 0].plot(time_steps[:boundary_idx], original_data[:boundary_idx, f_idx], label=input_label, color="blue", linewidth=1.5)
            line2 = axs[f_idx, 0].plot(time_steps[boundary_idx:], target_seq[:, f_idx], label=target_label, color="green", linestyle='--', linewidth=1.5)
            line3 = axs[f_idx, 0].plot(time_steps[boundary_idx:], pred_seq[:, f_idx], label=pred_label, color="orange", linestyle='-.', linewidth=1.5)
            
            axs[f_idx, 0].set_title(f"Feature {f_idx+1}")
            axs[f_idx, 0].set_xlabel("Time", fontsize=12)
            axs[f_idx, 0].set_ylabel("Value", fontsize=12)
            axs[f_idx, 0].grid(True, linestyle='--', alpha=0.6)

            if f_idx == 0:
                handles_left.extend(line1 + line2 + line3)
                labels_left.extend([obj.get_label() for obj in (line1 + line2 + line3)])

            sequence_label = "Data Sequence" if f_idx == 0 else "_nolegend_"
            split_label = "Val/Test Split" if f_idx == 0 else "_nolegend_"
            point_label = "Anomaly Points" if f_idx == 0 else "_nolegend_"
            
            line4 = axs[f_idx, 1].plot(time_steps[:boundary_idx], original_data[:boundary_idx, f_idx], label=sequence_label, color="blue", linewidth=1.5)

            line5 = axs[f_idx, 1].plot(time_steps[boundary_idx:], target_seq[:, f_idx], color="blue", linewidth=1.5)

            split_line = axs[f_idx, 1].axvline(x=boundary_idx, color="red", linestyle='--', linewidth=1, label=split_label)

            if isinstance(anomalies, dict):
                feature_key = f"Feature_{f_idx+1}"
                if feature_key in anomalies:
                    anomaly_indices = anomalies[feature_key]
                    scatter_plt = axs[f_idx, 1].scatter([time_steps[boundary_idx + i] for i in anomaly_indices], target_seq[anomaly_indices, f_idx], color="red", 
                        label=point_label, zorder=5)

            axs[f_idx, 1].set_title(f"Feature {f_idx+1}")
            axs[f_idx, 1].set_xlabel("Time", fontsize=12)
            axs[f_idx, 1].set_ylabel("Value", fontsize=12)
            axs[f_idx, 1].grid(True, linestyle='--', alpha=0.6)
            
            if f_idx == 0:
                handles_right.extend(line4)
                labels_right.extend([obj.get_label() for obj in line4])
                handles_right.append(split_line)
                labels_right.append(split_line.get_label())
                if scatter_plt is not None:
                    handles_right.append(scatter_plt)
                    labels_right.append(scatter_plt.get_label())

        plt.tight_layout()

        if handles_left:
            axs[0, 0].legend(handles_left, labels_left, loc="upper right", fontsize=10)
        if handles_right:
            unique_hr = dict(zip(labels_right, handles_right))
            axs[0, 1].legend(unique_hr.values(), unique_hr.keys(), loc="upper right", fontsize=10)

        plt.savefig(os.path.join(self.config["run_dir"], "anomalies_visualization.png"))
        plt.close()
