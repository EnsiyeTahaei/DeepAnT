import logging
import torch.nn as nn

# Set up logging
logger = logging.getLogger(__name__)


class DeepAntPredictor(nn.Module):
    def __init__(self, feature_dim, window_size, hidden_size=256):
        """
        DeepAnt predictor model.

        Args:
            feature_dim (int): Number of channels in the input data.
            window_size (int): Size of the sliding window.
            hidden_size (int): Number of hidden units in the fc layer (Defaults to 256).
        """
        super(DeepAntPredictor, self).__init__()

        self.model = nn.Sequential(
            nn.Conv1d(in_channels=feature_dim, out_channels=64, kernel_size=3, padding="valid"),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2),
            nn.Conv1d(in_channels=64, out_channels=128, kernel_size=1, padding="valid"),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2),
            nn.Flatten(),
            nn.Linear(in_features=(window_size - 2) // 4 * 128, out_features=hidden_size),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.25),
            nn.Linear(in_features=hidden_size, out_features=feature_dim),
        )

        logger.info("DeepAntPredictor model initialized.")

    def forward(self, x):
        """
        Forward pass of the DeepAnt model.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor after forward pass.
        """
        return self.model(x).squeeze(1)
