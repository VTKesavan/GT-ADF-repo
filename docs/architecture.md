# GT-ADF Architecture Documentation

## Overview

GT-ADF (Graph Transformer-based Anomaly Detection Framework) is designed to detect
cyber intrusions in smart grid EV charging networks. The framework processes
heterogeneous network traffic data as graphs and uses Graph Transformer layers to
capture both local and global dependencies for anomaly classification.

---

## Mathematical Foundations

### Node Feature Encoding (Eq. 5)

Each node (EV, charging station, grid component) is associated with a feature vector:

```
EV_n(gc_j, cs_i) = 1 - if(CC_j, OS_i, LC_i)
```

where `if` is the initial feature function combining:
- `CC_j`: Communication characteristics (protocol, port, message type)
- `OS_i`: Operating state (idle, charging, fault)
- `LC_i`: Location attributes (feeder, region)

### Graph Construction (Eq. 6)

Edges encode relationships between nodes via an adjacency/edge matrix:

```
ED_n = 0.5 * [μ(de) + μ(pf_tot)] * (EM_i, r_ij)
```

- `μ(de)`: Data-exchange relations (OCPP, ISO 15118 protocol flows)
- `μ(pf_tot)`: Total power-flow relations (grid-to-charger, charger-to-EV)
- `r_ij`: Relationship type between nodes i and j

### Graph Normalization (Eq. 7)

```
gc = Σ_i go*NA_i*Ec_i / Σ_i (RNC + ds_i) + hd_e(rg_j, nd_i)
```

This normalizes the graph object (`go`) with node attributes (`NA_i`) and edge
connections (`Ec_i`) to stabilize numeric computations.

---

## Graph Transformer Attention (Eqs. 1–4)

### Step 1: Feature Projection (Eq. 1)
```
q_i = W_q * x_i
k_i = W_k * x_i
v_i = W_v * x_i
```
where `W_q, W_k, W_v ∈ R^{d×d_k}` are learnable weight matrices.

### Step 2: Attention Scores (Eq. 2)
```
e_ij = (q_i^T k_j) / sqrt(d_k)
α_ij = exp(e_ij) / Σ_{l ∈ N(i)} exp(e_il)
```

### Step 3: Aggregation (Eq. 3)
```
h_i = Σ_{j ∈ N(i)} α_ij * v_j
```

### Step 4: Multi-Head Projection (Eq. 4)
```
h̃_i = Concat(h_i^(1), ..., h_i^(H)) * W_o
```
where H = 8 attention heads.

---

## Hybrid Semi-Supervised Loss

```
L_H = L_sup + λ * L_unsup
```

- **L_sup**: Cross-entropy on labeled nodes (20% of training data)
- **L_unsup**: KL divergence consistency between original and noise-perturbed predictions
- **λ = 0.5**: Balance weight

---

## Anomaly Scoring

Each node receives an anomaly score in [0, 100]:

```
score = sigmoid(W_score * h̃_i) * 100
```

Thresholds:
- score < T₁ (30)  → **Normal**
- T₁ ≤ score < T₂ (60) → **Suspicious**
- score ≥ T₂ (60) → **Anomaly**

---

## Hyperparameter Configuration

| Hyperparameter | Value | Justification |
|---------------|-------|---------------|
| Transformer layers | 4 | Ablation: best accuracy/efficiency trade-off |
| Attention heads | 8 | Captures diverse relationship types |
| Hidden dimension | 128 | Sufficient expressivity without overfitting |
| Dropout | 0.2 | Prevents overfitting on IDS datasets |
| Learning rate | 0.001 | Standard for Adam optimizer |
| Batch size | 64 | GPU memory vs. gradient stability |
| Early stopping | patience=10 | Based on validation loss |

---

## Complexity Analysis

- **Time complexity per layer**: O(|E| * d_k + N * d)
  where |E| = edges, N = nodes, d_k = head dimension, d = hidden dim
- **Space complexity**: O(N * d + |E| * H)
- **Parameters**: ~5.2M (comparable to GraphBERT at 6.8M)
- **Inference latency**: 16.8ms per batch (competitive with GNN at 18.6ms)

---

## Component Ablation Results (from paper)

| Configuration | Accuracy Drop |
|--------------|---------------|
| w/o Self-attention | −7 to −9% |
| w/o XAI module | slight recall drop |
| < 3 layers | underfitting |
| > 6 layers | diminishing returns |
| w/o edge features | −6% accuracy |
