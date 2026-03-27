# Sample Data (Synthetic)

This folder contains **synthetically generated** CSV files that mimic the
column structure and label distribution of each benchmark dataset.

> **These are NOT real network traffic captures.**
> They are randomly generated and will not reproduce paper results.
> Use them only to verify that the code runs correctly on your machine.

## Generating the files

```bash
python scripts/generate_sample_data.py
```

This takes approximately 5 seconds and produces:

| File | Rows | Columns | Purpose |
|------|------|---------|---------|
| `CICIDS2017_sample.csv` | 2,000 | 79 | Mimics CICIDS2017 MachineLearningCSV format |
| `UNSW_NB15_sample.csv`  | 2,000 | 44 | Mimics UNSW-NB15 training-set format |
| `ToN_IoT_sample.csv`    | 2,000 | 34 | Mimics ToN-IoT network traffic format |

## Using sample data

```bash
# Preprocess
python scripts/preprocess.py --dataset cicids2017 \
    --data_dir sample_data --output_dir data/processed --binary

# Train (5 epochs)
python scripts/train.py --config configs/cicids2017.yaml \
    --data_dir sample_data --max_epochs 5
```

## Why not include pre-generated files?

The generated files are excluded from git (via `.gitignore`) because:
- They contain no real scientific data — checking them in would mislead users.
- They are fast to regenerate locally.
- They would add ~2 MB of unnecessary binary-like CSV data to git history.
