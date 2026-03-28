# GT-ADF Architecture Documentation

## Overview

GT-ADF (Graph Transformer-based Anomaly Detection Framework) is designed to detect
cyber intrusions in smart grid EV charging networks. The framework processes
heterogeneous network traffic data as graphs and uses Graph Transformer layers to
capture both local and global dependencies for anomaly classification.

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
## Component Ablation Results

| Configuration | Accuracy Drop |
|--------------|---------------|
| w/o Self-attention | −7 to −9% |
| w/o XAI module | slight recall drop |
| < 3 layers | underfitting |
| > 6 layers | diminishing returns |
| w/o edge features | −6% accuracy |
