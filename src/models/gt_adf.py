"""
GT-ADF: Graph Transformer-based Anomaly Detection Framework
Main model definition.

"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import global_mean_pool, global_max_pool
from .graph_transformer import GraphTransformerLayer
from .attention import MultiHeadSelfAttention


class GTADF(nn.Module):
    """
    Graph Transformer-based Anomaly Detection Framework (GT-ADF).

    Architecture:
        1. Input feature projection
        2. Stacked Graph Transformer layers (multi-head self-attention)
        3. Global pooling
        4. Anomaly classification head (semi-supervised)
        5. Anomaly score generation

    Args:
        in_channels (int): Number of input node features.
        hidden_channels (int): Hidden embedding dimension (default: 128).
        out_channels (int): Number of output classes.
        num_layers (int): Number of Graph Transformer layers (default: 4).
        num_heads (int): Number of attention heads per layer (default: 8).
        dropout (float): Dropout probability (default: 0.2).
        lambda_unsup (float): Weight for unsupervised loss (default: 0.5).
    """

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int = 128,
        out_channels: int = 2,
        num_layers: int = 4,
        num_heads: int = 8,
        dropout: float = 0.2,
        lambda_unsup: float = 0.5,
    ):
        super(GTADF, self).__init__()

        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.out_channels = out_channels
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.dropout = dropout
        self.lambda_unsup = lambda_unsup

        # Input projection
        self.input_proj = nn.Linear(in_channels, hidden_channels)
        self.input_norm = nn.LayerNorm(hidden_channels)

        # Stacked Graph Transformer layers
        self.transformer_layers = nn.ModuleList([
            GraphTransformerLayer(
                in_channels=hidden_channels,
                out_channels=hidden_channels,
                num_heads=num_heads,
                dropout=dropout,
            )
            for _ in range(num_layers)
        ])

        # Layer normalization after each transformer block
        self.layer_norms = nn.ModuleList([
            nn.LayerNorm(hidden_channels) for _ in range(num_layers)
        ])

        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(hidden_channels * 2, hidden_channels),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels, out_channels),
        )

        # Anomaly score projection (for semi-supervised unsupervised term)
        self.score_proj = nn.Linear(hidden_channels, 1)

        self._init_weights()

    def _init_weights(self):
        """Xavier initialization for linear layers."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def encode(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through Graph Transformer encoder.

        Args:
            x: Node feature matrix [N, in_channels].
            edge_index: Graph connectivity [2, E].

        Returns:
            Node embeddings [N, hidden_channels].
        """
        # Project input features
        h = self.input_proj(x)
        h = self.input_norm(h)
        h = F.relu(h)

        # Pass through stacked transformer layers with residual connections
        for layer, norm in zip(self.transformer_layers, self.layer_norms):
            h_new = layer(h, edge_index)
            h = norm(h + h_new)  # Residual connection + layer norm
            h = F.dropout(h, p=self.dropout, training=self.training)

        return h

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        batch: torch.Tensor = None,
    ):
        """
        Full forward pass.

        Args:
            x: Node features [N, in_channels].
            edge_index: Graph edges [2, E].
            batch: Batch assignment vector [N] (for mini-batch graph training).

        Returns:
            logits: Class logits [B, out_channels].
            node_embeddings: Per-node embeddings [N, hidden_channels].
            anomaly_scores: Normalized anomaly scores [N, 1] in [0, 100].
        """
        # Encode: obtain node embeddings
        node_embeddings = self.encode(x, edge_index)

        # Graph-level pooling (mean + max concatenated)
        if batch is None:
            batch = torch.zeros(x.size(0), dtype=torch.long, device=x.device)

        graph_mean = global_mean_pool(node_embeddings, batch)
        graph_max = global_max_pool(node_embeddings, batch)
        graph_repr = torch.cat([graph_mean, graph_max], dim=-1)

        # Classification logits
        logits = self.classifier(graph_repr)

        # Per-node anomaly scores normalized to [0, 100]
        raw_scores = self.score_proj(node_embeddings)
        anomaly_scores = torch.sigmoid(raw_scores) * 100.0

        return logits, node_embeddings, anomaly_scores

    def classify_anomaly(
        self,
        anomaly_scores: torch.Tensor,
        t1: float = 30.0,
        t2: float = 60.0,
    ) -> torch.Tensor:
        """
        Convert anomaly scores to 3-class labels.

        Labels:
            0 = Normal   (score < T1)
            1 = Suspicious (T1 <= score < T2)
            2 = Anomaly  (score >= T2)

        Args:
            anomaly_scores: [N, 1] or [N] tensor.
            t1: Lower threshold (default: 30).
            t2: Upper threshold (default: 60).

        Returns:
            labels: [N] integer tensor.
        """
        scores = anomaly_scores.squeeze(-1)
        labels = torch.zeros_like(scores, dtype=torch.long)
        labels[scores >= t1] = 1
        labels[scores >= t2] = 2
        return labels

    def get_attention_weights(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
    ):
        """
        Extract attention weight matrices from all transformer layers.
        Used for XAI / interpretability analysis.

        Returns:
            List of attention weight tensors per layer.
        """
        attention_weights = []
        h = F.relu(self.input_norm(self.input_proj(x)))

        for layer, norm in zip(self.transformer_layers, self.layer_norms):
            h_new, attn = layer(h, edge_index, return_attention=True)
            attention_weights.append(attn)
            h = norm(h + h_new)

        return attention_weights


class HybridLoss(nn.Module):
    """
    Hybrid semi-supervised loss:
        L_H = L_sup + λ * L_unsup

    L_sup   = Cross-entropy on labeled nodes.
    L_unsup = Consistency loss between original and perturbed predictions.

    Args:
        lambda_unsup (float): Weight balancing supervised and unsupervised terms.
        num_classes (int): Number of output classes.
    """

    def __init__(self, lambda_unsup: float = 0.5, num_classes: int = 2):
        super(HybridLoss, self).__init__()
        self.lambda_unsup = lambda_unsup
        self.num_classes = num_classes
        self.ce_loss = nn.CrossEntropyLoss()
        self.kl_loss = nn.KLDivLoss(reduction="batchmean")

    def forward(
        self,
        logits_labeled: torch.Tensor,
        labels: torch.Tensor,
        logits_unlabeled: torch.Tensor = None,
        logits_perturbed: torch.Tensor = None,
    ) -> torch.Tensor:
        """
        Compute hybrid loss.

        Args:
            logits_labeled: Logits for labeled samples [N_L, C].
            labels: True labels for labeled samples [N_L].
            logits_unlabeled: Logits for unlabeled samples [N_U, C].
            logits_perturbed: Logits for perturbed unlabeled samples [N_U, C].

        Returns:
            Scalar hybrid loss.
        """
        # Supervised cross-entropy
        l_sup = self.ce_loss(logits_labeled, labels)

        # Unsupervised consistency loss (KL divergence)
        if logits_unlabeled is not None and logits_perturbed is not None:
            p_unlabeled = F.softmax(logits_unlabeled, dim=-1)
            p_perturbed = F.log_softmax(logits_perturbed, dim=-1)
            l_unsup = self.kl_loss(p_perturbed, p_unlabeled)
        else:
            l_unsup = torch.tensor(0.0, device=logits_labeled.device)

        return l_sup + self.lambda_unsup * l_unsup
