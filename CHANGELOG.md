# Changelog

All notable changes to GT-ADF are documented in this file.
Format follows [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).

---

## [1.0.0] — 2025

### Added
- Initial public release of GT-ADF framework
- `GraphTransformerLayer` implementing Equations 1–4 (multi-head self-attention via PyG `MessagePassing`)
- `GTADF` model: 4-layer Graph Transformer encoder + semi-supervised anomaly detection head
- `HybridLoss`: supervised cross-entropy + unsupervised KL-divergence consistency loss
- `MultiHeadSelfAttention`: standalone reference implementation (non-PyG)
- Baseline models: GNN (GCN), CNN-IDS, LSTM-IDS, SVM, Random Forest
- Dataset loaders for CICIDS2017-SafeML, UNSW-NB15, and ToN-IoT
- `Preprocessor`: Min-Max scaling, variance filtering, correlation filtering, categorical encoding
- `GraphBuilder`: k-NN and sequential graph construction from tabular IDS data
- `build_ev_topology_graph()`: explicit EV/CS/GC heterogeneous graph (Eqs. 5–7)
- `Trainer` with Adam optimiser, learning-rate scheduler, early stopping, TensorBoard logging
- Full evaluation suite: Accuracy, Precision, Recall, F1, AUC, MCC, NPV, confusion matrix
- All paper figures (Figures 7–14) as reproducible plot functions
- CLI scripts: `preprocess.py`, `train.py`, `evaluate.py`, `run_baselines.py`
- `generate_sample_data.py`: synthetic data generator for quick-start testing
- 4 Jupyter notebooks (data exploration, graph construction, training, visualisation)
- 3 pytest test files (model, metrics, dataset/graph-builder)
- GitHub Actions CI (CPU-only, Python 3.9 and 3.10)
- YAML configs for all three datasets
- `environment.yml` (conda), `Dockerfile` (CPU), `INSTALL.md`
- `CITATION.cff`, `CONTRIBUTING.md`, `CHANGELOG.md`

### Known limitations
- No pretrained checkpoint is distributed (model must be trained from scratch)
- Synthetic sample data does not reproduce paper metrics
- Real hardware deployment on live EV charging testbeds is not yet validated

---

## [Unreleased]

### Planned
- Federated GT-ADF (each EV station trains locally, FedAvg aggregation)
- Deep RL-based adaptive mitigation agent
- TGN (Temporal Graph Network) extension for streaming EV sessions
- Integration with a smart-grid digital-twin testbed
- Pretrained checkpoint release on Zenodo or HuggingFace Hub
