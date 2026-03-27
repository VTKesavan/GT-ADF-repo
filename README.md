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
> dataset is publicly available. 

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

---


