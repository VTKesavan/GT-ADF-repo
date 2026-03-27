# GT-ADF: Graph Transformer-Based Anomaly Detection Framework for Smart Grid EV Charging Stations

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-orange)](https://pytorch.org/)
[![PyG](https://img.shields.io/badge/PyG-2.3%2B-green)](https://pytorch-geometric.readthedocs.io/)
[![CI](https://github.com/your-username/GT-ADF/actions/workflows/ci.yml/badge.svg)](https://github.com/your-username/GT-ADF/actions)
[![License](https://img.shields.io/badge/License-MIT-yellow)](LICENSE)

> **Paper:** *Securing Smart Grid EV Charging Stations using Graph Transformer-Based Anomaly Detection*
> **Authors:** [Thiruppathy Kesavan V, Gopi R, Md. Jakir Hossen, Danalakshmi D, Emerson Raja Joseph]
> 

---

## Overview

This repository is the official implementation of **GT-ADF** — a Graph Transformer-based
Anomaly Detection Framework for securing smart grid Electric Vehicle (EV) charging stations
against cyber threats.

GT-ADF uses multi-head self-attention over graph-structured network data to capture both
local and global dependencies simultaneously, achieving state-of-the-art intrusion detection
across three benchmark datasets.

### Key results (CICIDS2017-SafeML)

| Model | Accuracy | Precision | Recall | F1-Score | AUC |
|---|---|---|---|---|---|
| SVM | 85.2% | 84.6% | 83.1% | 84.6% | 86.1% |
| Random Forest | 88.7% | 87.9% | 86.5% | 87.9% | 89.3% |
| CNN-IDS | 90.4% | 89.8% | 88.4% | 89.8% | 91.2% |
| LSTM | 91.6% | 90.9% | 89.7% | 90.9% | 92.5% |
| GNN | 92.8% | 92.1% | 91.3% | 92.1% | 93.6% |
| TADDY (2023) | 94.1% | 93.5% | 92.8% | 93.5% | 94.8% |
| GraphBERT | 94.8% | 94.2% | 93.6% | 94.2% | 95.3% |
| Federated GNN-IDS (2024) | 93.6% | 92.9% | 92.2% | 92.9% | 94.1% |
| **GT-ADF (ours)** | **96.2%** | **97.0%** | **95.0%** | **95.7%** | **96.8%** |

Full results for UNSW-NB15 and ToN-IoT: [`results/sample_outputs/reported_results.json`](results/sample_outputs/reported_results.json)

---

## Architecture

```
Raw network traffic / EV charging session data
            ↓
    Data preprocessing
    (Min-Max scaling, encoding, feature selection)
            ↓
    Graph construction  [Eqs. 5-6]
    Nodes: EVs · Charging stations · Grid components
    Edges: Data exchange · Power flow · Auth logs
            ↓
    4 x Graph Transformer layers  [Eqs. 1-4]
    (8 attention heads, hidden dim = 128)
            ↓
    Semi-supervised anomaly detection
    Hybrid loss: L_H = L_sup + lambda * L_unsup
            ↓
    Anomaly score [0,100]  +  XAI explanation
    Normal (<30)  Suspicious (30-60)  Anomaly (>=60)
```

For full mathematical derivations see [`docs/architecture.md`](docs/architecture.md).

---

## Installation

See [`INSTALL.md`](INSTALL.md) for full instructions (GPU/CPU, conda, Docker).

**Quick install (CPU, pip):**
```bash
git clone https://github.com/your-username/GT-ADF.git
cd GT-ADF
python -m venv venv && source venv/bin/activate
pip install torch==2.0.1+cpu --index-url https://download.pytorch.org/whl/cpu
pip install torch-geometric
pip install torch-scatter torch-sparse torch-cluster \
    -f https://data.pyg.org/whl/torch-2.0.1+cpu.html
pip install -r requirements.txt && pip install -e .
```

---

## Datasets

Datasets are not included in this repository — download them separately.

| Dataset | Official download | Kaggle mirror | Place in |
|---------|------------------|---------------|----------|
| CICIDS2017-SafeML | [CIC server](http://205.174.165.80/CICDataset/CIC-IDS-2017/Dataset/MachineLearningCSV.zip) | [Kaggle](https://www.kaggle.com/datasets/cicdataset/cicids2017) | `data/raw/CICIDS2017/` |
| UNSW-NB15 | [UNSW](https://research.unsw.edu.au/projects/unsw-nb15-dataset) | [Kaggle](https://www.kaggle.com/datasets/dhoogla/unswnb15) | `data/raw/UNSW_NB15/` |
| ToN-IoT | [UNSW](https://research.unsw.edu.au/projects/toniot-datasets) | [Kaggle](https://www.kaggle.com/datasets/arnobbhowmik/ton-iot-network-dataset) | `data/raw/ToN_IoT/` |

Each subfolder has a `README.md` with exact file placement instructions.

> **Why these datasets?** No large-scale labeled EV-charging-specific cybersecurity
> dataset is publicly available. These benchmark IDS datasets represent diverse,
> high-quality modern network attacks and allow standardized comparison with prior work.
> This limitation is discussed in Section 5 of the paper.

---

## Quick start — no download required

```bash
# Generate synthetic data (mimics dataset structure, 2000 rows each, ~5 sec)
python scripts/generate_sample_data.py

# Preprocess into graphs
python scripts/preprocess.py --dataset cicids2017 \
    --data_dir sample_data --output_dir data/processed --binary

# Run unit tests
pytest tests/ -v

# Short training run to verify the pipeline (5 epochs)
python scripts/train.py --config configs/cicids2017.yaml \
    --data_dir sample_data --max_epochs 5
```

> Synthetic data will not reproduce paper results — it only verifies the pipeline runs.

---

## Full experiment pipeline

```bash
# Preprocess
python scripts/preprocess.py --dataset cicids2017 \
    --data_dir data/raw/CICIDS2017 --output_dir data/processed

# Train
python scripts/train.py --config configs/cicids2017.yaml

# Evaluate
python scripts/evaluate.py \
    --checkpoint results/cicids2017/checkpoints/gt_adf_best.pt \
    --config configs/cicids2017.yaml

# Run all baselines
python scripts/run_baselines.py --config configs/cicids2017.yaml
```

---

## Hyperparameters

| Parameter | Value |
|-----------|-------|
| Transformer layers | 4 |
| Attention heads (H) | 8 |
| Hidden dimension | 128 |
| Dropout | 0.2 |
| Optimizer | Adam, lr = 0.001 |
| Batch size | 64 |
| Max epochs | 100 (early stopping, patience = 10) |
| Labeled ratio | 20% of training nodes |
| lambda_unsup | 0.5 |
| k-NN neighbors | 5 |
| Window size | 10 records/graph |

---

## Project structure

```
GT-ADF/
├── README.md
├── INSTALL.md
├── CITATION.cff
├── requirements.txt / environment.yml / Dockerfile
├── configs/                     YAML configs per dataset
├── data/
│   ├── raw/CICIDS2017/README.md    download instructions
│   ├── raw/UNSW_NB15/README.md
│   └── raw/ToN_IoT/README.md
├── sample_data/                 auto-generated synthetic data
├── src/
│   ├── models/gt_adf.py         GT-ADF model + HybridLoss
│   ├── models/graph_transformer.py
│   ├── models/attention.py
│   ├── models/baselines.py      GNN, CNN, LSTM, SVM, RF
│   ├── data/dataset.py          dataset loaders
│   ├── data/preprocessor.py
│   ├── data/graph_builder.py
│   ├── training/trainer.py
│   ├── evaluation/metrics.py
│   └── visualization/plots.py   all paper figures
├── scripts/
│   ├── generate_sample_data.py
│   ├── preprocess.py
│   ├── train.py
│   ├── evaluate.py
│   └── run_baselines.py
├── notebooks/                   4 Jupyter notebooks
├── tests/                       pytest unit tests
├── results/sample_outputs/reported_results.json
└── docs/architecture.md
```

---

## Citation

```bibtex
@article{gtadf2026,
  title   = {Securing Smart Grid EV Charging Stations using Graph Transformer-Based Anomaly Detection},
  author  = {Thiruppathy Kesavan V, Gopi R, Md. Jakir Hossen, Danalakshmi D, Emerson Raja Joseph},
  journal = {},
  year    = {2026}
}
```

---

## Limitations

- Results are from benchmark IDS datasets, not EV-charging-specific data (none is publicly available at scale).
- Experiments are simulation-based; real hardware deployment is planned as future work.
- Latency figures (Table 5) are hardware-specific (Intel Core i7, 16GB RAM, NVIDIA GPU).

---


