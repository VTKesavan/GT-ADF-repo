"""
Baseline models for comparison with GT-ADF.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
import numpy as np


# -----------------------------------------------------------------------
# GNN Baseline (GCN)
# -----------------------------------------------------------------------

class GNNBaseline(nn.Module):
    """
    Standard Graph Convolutional Network baseline.

    Uses 3 GCN layers with ReLU activations and mean global pooling.
    """

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int = 128,
        out_channels: int = 2,
        num_layers: int = 3,
        dropout: float = 0.2,
    ):
        super(GNNBaseline, self).__init__()
        self.convs = nn.ModuleList()
        self.convs.append(GCNConv(in_channels, hidden_channels))
        for _ in range(num_layers - 2):
            self.convs.append(GCNConv(hidden_channels, hidden_channels))
        self.convs.append(GCNConv(hidden_channels, hidden_channels))

        self.classifier = nn.Linear(hidden_channels, out_channels)
        self.dropout = dropout

    def forward(self, x, edge_index, batch=None):
        for conv in self.convs[:-1]:
            x = F.relu(conv(x, edge_index))
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.convs[-1](x, edge_index)

        if batch is None:
            batch = torch.zeros(x.size(0), dtype=torch.long, device=x.device)
        x = global_mean_pool(x, batch)

        return self.classifier(x)


# -----------------------------------------------------------------------
# CNN-IDS Baseline
# -----------------------------------------------------------------------

class CNNBaseline(nn.Module):
    """
    1D Convolutional Neural Network for network intrusion detection.
    Treats node feature vectors as 1D signals.
    """

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int = 128,
        out_channels: int = 2,
        dropout: float = 0.2,
    ):
        super(CNNBaseline, self).__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(hidden_channels // 4),
        )
        self.fc = nn.Sequential(
            nn.Linear(64 * (hidden_channels // 4), hidden_channels),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels, out_channels),
        )
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels

    def forward(self, x, edge_index=None, batch=None):
        # x: [N, F] — treat each sample independently
        x = x.unsqueeze(1)  # [N, 1, F]
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)


# -----------------------------------------------------------------------
# LSTM-IDS Baseline
# -----------------------------------------------------------------------

class LSTMBaseline(nn.Module):
    """
    LSTM-based intrusion detection system.
    Treats each sample as a sequence of features.
    """

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int = 128,
        out_channels: int = 2,
        num_layers: int = 2,
        dropout: float = 0.2,
    ):
        super(LSTMBaseline, self).__init__()
        self.lstm = nn.LSTM(
            input_size=1,
            hidden_size=hidden_channels,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.classifier = nn.Linear(hidden_channels, out_channels)

    def forward(self, x, edge_index=None, batch=None):
        # x: [N, F] → treat features as timesteps: [N, F, 1]
        x = x.unsqueeze(-1)  # [N, F, 1]
        out, (h_n, _) = self.lstm(x)
        return self.classifier(h_n[-1])  # last hidden state


# -----------------------------------------------------------------------
# Sklearn-based Baselines (SVM, Random Forest)
# -----------------------------------------------------------------------

class SVMBaseline:
    """
    SVM-based IDS wrapper for sklearn.

    Usage:
        svm = SVMBaseline()
        svm.fit(X_train, y_train)
        preds = svm.predict(X_test)
        scores = svm.predict_proba(X_test)
    """

    def __init__(self, kernel: str = "rbf", C: float = 1.0, probability: bool = True):
        self.model = SVC(kernel=kernel, C=C, probability=probability)

    def fit(self, X: np.ndarray, y: np.ndarray):
        self.model.fit(X, y)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict_proba(X)


class RandomForestBaseline:
    """
    Random Forest IDS wrapper.

    Usage:
        rf = RandomForestBaseline()
        rf.fit(X_train, y_train)
        preds = rf.predict(X_test)
    """

    def __init__(self, n_estimators: int = 100, max_depth: int = None, n_jobs: int = -1):
        self.model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            n_jobs=n_jobs,
            random_state=42,
        )

    def fit(self, X: np.ndarray, y: np.ndarray):
        self.model.fit(X, y)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict_proba(X)


# -----------------------------------------------------------------------
# Model Registry
# -----------------------------------------------------------------------

BASELINE_REGISTRY = {
    "gnn": GNNBaseline,
    "cnn": CNNBaseline,
    "lstm": LSTMBaseline,
    "svm": SVMBaseline,
    "rf": RandomForestBaseline,
}


def get_baseline(name: str, **kwargs):
    """Factory function to instantiate a baseline by name."""
    name_lower = name.lower()
    if name_lower not in BASELINE_REGISTRY:
        raise ValueError(
            f"Unknown baseline '{name}'. "
            f"Available: {list(BASELINE_REGISTRY.keys())}"
        )
    return BASELINE_REGISTRY[name_lower](**kwargs)
