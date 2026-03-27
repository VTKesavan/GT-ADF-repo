#!/usr/bin/env python3
"""
Evaluate a saved GT-ADF checkpoint on a test dataset.

Usage:
    python scripts/evaluate.py --checkpoint results/checkpoints/gt_adf_best.pt \
                                --config configs/cicids2017.yaml
"""

import sys
import os
import argparse
import json
import logging

import yaml
import torch
from torch_geometric.loader import DataLoader

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data.dataset import load_dataset
from src.models.gt_adf import GTADF
from src.evaluation.metrics import evaluate_model, print_metrics_table
from src.utils.helpers import set_seed, get_device, setup_logging, load_checkpoint

logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate GT-ADF checkpoint")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--dataset", type=str, default=None, help="Override dataset name")
    parser.add_argument("--data_dir", type=str, default=None, help="Override data dir")
    return parser.parse_args()


def main():
    args = parse_args()
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    setup_logging()
    set_seed(config.get("seed", 42))
    device = get_device(prefer_gpu=config.get("use_gpu", True))

    dataset_name = args.dataset or config["dataset"]
    data_dir = args.data_dir or config["data_dir"]

    test_dataset = load_dataset(
        dataset_name=dataset_name,
        data_dir=data_dir,
        split="test",
        binary=config.get("binary", False),
        seed=config.get("seed", 42),
    )
    test_loader = DataLoader(test_dataset, batch_size=config.get("batch_size", 64))

    sample = test_dataset[0]
    in_channels = sample.x.size(-1)
    num_classes = config.get("num_classes", 2)

    model = GTADF(
        in_channels=in_channels,
        hidden_channels=config.get("hidden_dim", 128),
        out_channels=num_classes,
        num_layers=config.get("num_layers", 4),
        num_heads=config.get("num_heads", 8),
        dropout=config.get("dropout", 0.2),
    )

    load_checkpoint(model, args.checkpoint, device=device)

    metrics = evaluate_model(model, test_loader, device=device)
    print_metrics_table(metrics, dataset_name=dataset_name.upper())

    out_path = os.path.join(
        os.path.dirname(args.checkpoint), f"eval_{dataset_name}.json"
    )
    with open(out_path, "w") as f:
        json.dump({k: v for k, v in metrics.items() if k != "confusion_matrix"}, f, indent=2)
    logger.info(f"Results saved to {out_path}")


if __name__ == "__main__":
    main()
