"""
Standalone multi-head self-attention reference for GT-ADF.

This module provides a pure-PyTorch (non-PyG) implementation of the
attention mechanism described in Equations (1)–(4) of the paper,
useful for testing, ablation, and educational reference.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class ScaledDotProductAttention(nn.Module):
    """
    Scaled dot-product attention (Eq. 2):
        e_ij = q_i^T k_j / sqrt(d_k)
        alpha_ij = softmax(e_ij) over N(i)
        h_i = sum_j alpha_ij * v_j

    Args:
        dropout (float): Attention weight dropout probability.
    """

    def __init__(self, dropout: float = 0.0):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        Q: torch.Tensor,
        K: torch.Tensor,
        V: torch.Tensor,
        mask: torch.Tensor = None,
    ):
        """
        Args:
            Q: Query tensor [B, H, N, d_k]
            K: Key tensor   [B, H, N, d_k]
            V: Value tensor [B, H, N, d_v]
            mask: Optional boolean mask [B, 1, N, N] (True = ignore)

        Returns:
            output: [B, H, N, d_v]
            attn_weights: [B, H, N, N]
        """
        d_k = Q.size(-1)
        # Unnormalized attention scores: e_ij
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(d_k)  # [B,H,N,N]

        if mask is not None:
            scores = scores.masked_fill(mask, float("-inf"))

        # Normalized attention coefficients: alpha_ij
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # Weighted sum: h_i = sum_j alpha_ij * v_j
        output = torch.matmul(attn_weights, V)  # [B, H, N, d_v]
        return output, attn_weights


class MultiHeadSelfAttention(nn.Module):
    """
    Multi-head self-attention for node embeddings (Eqs. 1–4).

    Given node feature matrix X ∈ R^{N×d}:
        q_i = W_q x_i,  k_i = W_k x_i,  v_i = W_v x_i
        h̃_i = Concat(h_i^(1), ..., h_i^(H)) W_o

    Args:
        embed_dim (int): Total embedding dimension d.
        num_heads (int): Number of parallel heads H.
        dropout (float): Attention dropout.
        bias (bool): Whether to use bias in projections.
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int = 8,
        dropout: float = 0.0,
        bias: bool = True,
    ):
        super().__init__()
        assert embed_dim % num_heads == 0

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        # Projection matrices W_q, W_k, W_v (Eq. 1)
        self.W_q = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.W_k = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.W_v = nn.Linear(embed_dim, embed_dim, bias=bias)

        # Output projection W_o (Eq. 4)
        self.W_o = nn.Linear(embed_dim, embed_dim, bias=bias)

        self.attention = ScaledDotProductAttention(dropout=dropout)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor = None,
        return_weights: bool = False,
    ):
        """
        Args:
            x: Node feature matrix [N, d] or batched [B, N, d].
            mask: Optional attention mask.
            return_weights: If True, also return attention weight tensor.

        Returns:
            out: Updated node embeddings [N, d] or [B, N, d].
            attn_weights (optional): [B, H, N, N] or [H, N, N].
        """
        unbatched = x.dim() == 2
        if unbatched:
            x = x.unsqueeze(0)  # [1, N, d]

        B, N, d = x.shape

        # Compute Q, K, V (Eq. 1)
        Q = self.W_q(x).view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.W_k(x).view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.W_v(x).view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        # Shape: [B, H, N, d_k]

        # Scaled dot-product attention (Eqs. 2–3)
        attn_out, attn_weights = self.attention(Q, K, V, mask=mask)

        # Concatenate heads and project (Eq. 4)
        attn_out = attn_out.transpose(1, 2).contiguous().view(B, N, d)
        out = self.W_o(attn_out)

        # Residual + layer norm
        out = self.norm(x + self.dropout(out))

        if unbatched:
            out = out.squeeze(0)
            attn_weights = attn_weights.squeeze(0)

        if return_weights:
            return out, attn_weights
        return out
