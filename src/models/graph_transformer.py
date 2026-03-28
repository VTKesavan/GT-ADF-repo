"""
Graph Transformer Layer with multi-head self-attention.

"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import softmax


class GraphTransformerLayer(MessagePassing):
    """
    Single Graph Transformer layer implementing scaled dot-product
    multi-head self-attention over a graph.

    Args:
        in_channels (int): Input feature dimension.
        out_channels (int): Output feature dimension.
        num_heads (int): Number of parallel attention heads.
        dropout (float): Attention dropout probability.
        bias (bool): Whether to use bias in linear projections.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        num_heads: int = 8,
        dropout: float = 0.2,
        bias: bool = True,
    ):
        super(GraphTransformerLayer, self).__init__(aggr="add", node_dim=0)

        assert out_channels % num_heads == 0, (
            f"out_channels ({out_channels}) must be divisible by num_heads ({num_heads})"
        )

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_heads = num_heads
        self.head_dim = out_channels // num_heads
        self.dropout = dropout
        self.scale = math.sqrt(self.head_dim)

        # Projection matrices: W_q, W_k, W_v
        self.W_q = nn.Linear(in_channels, out_channels, bias=bias)
        self.W_k = nn.Linear(in_channels, out_channels, bias=bias)
        self.W_v = nn.Linear(in_channels, out_channels, bias=bias)

        # Output projection W_o
        self.W_o = nn.Linear(out_channels, out_channels, bias=bias)

        # Feed-forward sub-layer (after attention)
        self.ffn = nn.Sequential(
            nn.Linear(out_channels, out_channels * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(out_channels * 2, out_channels),
        )

        self.norm1 = nn.LayerNorm(out_channels)
        self.norm2 = nn.LayerNorm(out_channels)
        self.dropout_layer = nn.Dropout(dropout)

        self._stored_attention = None

    def forward(
        self,
        x: Tensor,
        edge_index: Tensor,
        return_attention: bool = False,
    ):
        """
        Args:
            x: Node features [N, in_channels].
            edge_index: Graph connectivity [2, E].
            return_attention: If True, also return attention weights.

        Returns:
            out: Updated node embeddings [N, out_channels].
            attn (optional): Attention weights [E, num_heads].
        """
        # Compute query, key, value projections
        Q = self.W_q(x).view(-1, self.num_heads, self.head_dim)  # [N, H, d_k]
        K = self.W_k(x).view(-1, self.num_heads, self.head_dim)  # [N, H, d_k]
        V = self.W_v(x).view(-1, self.num_heads, self.head_dim)  # [N, H, d_k]

        # Propagate: compute attention-weighted aggregation
        out = self.propagate(edge_index, Q=Q, K=K, V=V, num_nodes=x.size(0))

        # Reshape and apply output projection
        out = out.view(-1, self.out_channels)
        out = self.W_o(out)
        out = self.dropout_layer(out)

        # Feed-forward sub-layer with residual
        out = self.norm1(out)
        out = out + self.ffn(out)
        out = self.norm2(out)

        if return_attention:
            return out, self._stored_attention
        return out

    def message(self, Q_i: Tensor, K_j: Tensor, V_j: Tensor, index: Tensor):
        """
        Compute attention-weighted messages from neighbors j to node i.

        Equations (1)-(3) from the paper:
            e_ij = (q_i^T k_j) / sqrt(d_k)
            alpha_ij = softmax over N(i)
            h_i = sum_j alpha_ij * v_j
        """
        # Unnormalized attention score: e_ij = q_i^T k_j / sqrt(d_k)
        attn_scores = (Q_i * K_j).sum(dim=-1, keepdim=True) / self.scale  # [E, H, 1]

        # Normalize attention coefficients over each node's neighborhood
        attn_weights = softmax(attn_scores, index)  # [E, H, 1]
        attn_weights = F.dropout(attn_weights, p=self.dropout, training=self.training)

        # Store for XAI extraction
        self._stored_attention = attn_weights.squeeze(-1).detach()  # [E, H]

        # Weighted sum: h_i = sum_j alpha_ij * v_j
        return attn_weights * V_j  # [E, H, d_k]

    def update(self, aggr_out: Tensor):
        """Return aggregated messages unchanged."""
        return aggr_out

    def __repr__(self):
        return (
            f"{self.__class__.__name__}("
            f"in={self.in_channels}, out={self.out_channels}, "
            f"heads={self.num_heads}, dropout={self.dropout})"
        )
