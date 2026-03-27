# Installation Guide

This page gives detailed step-by-step instructions for every supported environment.
PyTorch Geometric (PyG) requires platform-specific wheels — this is the main
complexity. Follow the block that matches your setup.

---

## Option A — pip + venv (recommended for most users)

```bash
# 1. Create virtual environment
python -m venv venv
source venv/bin/activate        # Linux / macOS
venv\Scripts\activate           # Windows

# 2. Install PyTorch
#    ── GPU (CUDA 11.8) ──────────────────────────────────────────────
pip install torch==2.0.1 torchvision==0.15.2 \
    --index-url https://download.pytorch.org/whl/cu118
#    ── GPU (CUDA 12.1) ──────────────────────────────────────────────
# pip install torch==2.0.1 torchvision==0.15.2 \
#     --index-url https://download.pytorch.org/whl/cu121
#    ── CPU only ─────────────────────────────────────────────────────
# pip install torch==2.0.1 torchvision==0.15.2 \
#     --index-url https://download.pytorch.org/whl/cpu

# 3. Install PyTorch Geometric
#    ── GPU (CUDA 11.8) ──────────────────────────────────────────────
pip install torch-geometric
pip install torch-scatter torch-sparse torch-cluster torch-spline-conv \
    -f https://data.pyg.org/whl/torch-2.0.1+cu118.html
#    ── CPU only ─────────────────────────────────────────────────────
# pip install torch-geometric
# pip install torch-scatter torch-sparse torch-cluster \
#     -f https://data.pyg.org/whl/torch-2.0.1+cpu.html

# 4. Install remaining dependencies
pip install -r requirements.txt

# 5. Install the GT-ADF package (editable)
pip install -e .
```

---

## Option B — Conda

```bash
# 1. Create environment (GPU build — edit environment.yml for CPU)
conda env create -f environment.yml

# 2. Activate
conda activate gt-adf

# 3. Install PyTorch Geometric via pip inside the conda env
#    (replace cu118 with your CUDA version)
pip install torch-geometric
pip install torch-scatter torch-sparse torch-cluster \
    -f https://data.pyg.org/whl/torch-2.0.1+cu118.html

# 4. Install GT-ADF
pip install -e .
```

---

## Option C — Docker (CPU)

```bash
docker build -t gt-adf .
docker run --rm -it -v $(pwd)/data:/workspace/data gt-adf bash
```

---

## Verify the installation

```bash
python -c "
import torch
import torch_geometric
from src.models.gt_adf import GTADF
print(f'PyTorch:          {torch.__version__}')
print(f'PyG:              {torch_geometric.__version__}')
print(f'CUDA available:   {torch.cuda.is_available()}')
model = GTADF(in_channels=20, hidden_channels=32, out_channels=2,
              num_layers=2, num_heads=4)
print(f'GT-ADF model:     OK ({sum(p.numel() for p in model.parameters()):,} params)')
"
```

Expected output (CPU example):
```
PyTorch:          2.0.1
PyG:              2.3.1
CUDA available:   False
GT-ADF model:     OK (397,314 params)
```

---

## Quick-start without downloading datasets

Run a full pipeline on the built-in synthetic data:

```bash
# Generate synthetic data (2000 rows per dataset, ~5 seconds)
python scripts/generate_sample_data.py

# Preprocess (converts CSV → graph objects)
python scripts/preprocess.py --dataset cicids2017 \
    --data_dir sample_data --output_dir data/processed --binary

# Train for 5 epochs to verify the pipeline
python scripts/train.py --config configs/cicids2017.yaml \
    --data_dir sample_data --max_epochs 5
```

The synthetic data does not reproduce paper results — it is only for
verifying that the code runs end-to-end on your machine.

---

## Known issues

| Issue | Fix |
|-------|-----|
| `ImportError: torch_scatter` | Install from PyG wheels, not PyPI — see step 3 above |
| `CUDA error: no kernel image` | Your CUDA version doesn't match the wheel — rerun step 3 with the correct suffix |
| Segfault on macOS with MPS | Set `use_gpu: false` in your config YAML |
| `FileNotFoundError` in dataset loader | Check that CSV files are placed in the correct subfolder — see `data/raw/*/README.md` |
