#!/usr/bin/env python3
"""
Main training script for GT-ADF.

Usage:
    python scripts/train.py --config configs/cicids2017.yaml
    python scripts/train.py --config configs/unsw_nb15.yaml --seed 123
"""

import sys
import os
import argparse
import logging
import json

import yaml
import torch
from torch_geometric.loader import DataLoader

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data.dataset import load_dataset
from src.models.gt_adf import GTADF
from src.training.trainer import Trainer
from src.evaluation.metrics import evaluate_model, print_metrics_table
from src.visualization.plots import generate_all_figures
from src.utils.helpers import set_seed, get_device, setup_logging, model_summary

logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="Train GT-ADF")
    parser.add_argument(
        "--config", type=str, required=True,
        help="Path to YAML config file (e.g., configs/cicids2017.yaml)"
    )
    parser.add_argument("--seed", type=int, default=None, help="Override random seed")
    parser.add_argument("--gpu", action="store_true", help="Force GPU usage")
    parser.add_argument("--no_train", action="store_true", help="Skip training, evaluate only")
    # CLI overrides so quick-start / CI can point at sample_data
    parser.add_argument("--data_dir", type=str, default=None,
                        help="Override data_dir from config (useful for sample_data)")
    parser.add_argument("--max_epochs", type=int, default=None,
                        help="Override max_epochs from config")
    return parser.parse_args()


def load_config(path: str) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def main():
    args = parse_args()
    config = load_config(args.config)

    if args.seed is not None:
        config["seed"] = args.seed
    if args.data_dir is not None:
        config["data_dir"] = args.data_dir
    if args.max_epochs is not None:
        config["max_epochs"] = args.max_epochs

    # Setup
    setup_logging(
        level=config.get("log_level", "INFO"),
        log_file=os.path.join(config.get("output_dir", "results"), "logs", "train.log"),
    )
    set_seed(config.get("seed", 42))
    device = get_device(prefer_gpu=args.gpu or config.get("use_gpu", True))

    logger.info(f"Config: {json.dumps(config, indent=2)}")

    # ----------------------------------------------------------------
    # Load Datasets
    # ----------------------------------------------------------------
    dataset_name = config["dataset"]
    data_dir = config["data_dir"]

    logger.info(f"Loading {dataset_name} dataset from {data_dir}")

    train_dataset = load_dataset(
        dataset_name=dataset_name,
        data_dir=data_dir,
        split="train",
        binary=config.get("binary", False),
        seed=config.get("seed", 42),
    )
    test_dataset = load_dataset(
        dataset_name=dataset_name,
        data_dir=data_dir,
        split="test",
        binary=config.get("binary", False),
        seed=config.get("seed", 42),
    )

    logger.info(f"Train graphs: {len(train_dataset)} | Test graphs: {len(test_dataset)}")

    # Split train into train/val (80/20 of train portion)
    val_size = max(1, int(len(train_dataset) * 0.2))
    train_size = len(train_dataset) - val_size
    train_subset, val_subset = torch.utils.data.random_split(
        train_dataset, [train_size, val_size]
    )

    train_loader = DataLoader(
        train_subset,
        batch_size=config.get("batch_size", 64),
        shuffle=True,
        num_workers=config.get("num_workers", 2),
    )
    val_loader = DataLoader(
        val_subset,
        batch_size=config.get("batch_size", 64),
        shuffle=False,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=config.get("batch_size", 64),
        shuffle=False,
    )

    # ----------------------------------------------------------------
    # Build Model
    # ----------------------------------------------------------------
    sample = train_dataset[0]
    in_channels = sample.x.size(-1)
    num_classes = config.get("num_classes", 2)

    model = GTADF(
        in_channels=in_channels,
        hidden_channels=config.get("hidden_dim", 128),
        out_channels=num_classes,
        num_layers=config.get("num_layers", 4),
        num_heads=config.get("num_heads", 8),
        dropout=config.get("dropout", 0.2),
        lambda_unsup=config.get("lambda_unsup", 0.5),
    )

    model_summary(model)

    # ----------------------------------------------------------------
    # Train
    # ----------------------------------------------------------------
    output_dir = config.get("output_dir", "results")
    trainer = Trainer(model=model, config=config, device=device, output_dir=output_dir)

    if not args.no_train:
        history = trainer.train(
            train_loader=train_loader,
            val_loader=val_loader,
            max_epochs=config.get("max_epochs", 100),
            patience=config.get("patience", 10),
            labeled_ratio=config.get("labeled_ratio", 0.2),
        )
        # Save history
        history_path = os.path.join(output_dir, "train_history.json")
        with open(history_path, "w") as f:
            json.dump(history, f, indent=2)
        logger.info(f"Training history saved to {history_path}")

    # ----------------------------------------------------------------
    # Evaluate on Test Set
    # ----------------------------------------------------------------
    best_ckpt = os.path.join(output_dir, "checkpoints", "gt_adf_best.pt")
    if os.path.exists(best_ckpt):
        from src.utils.helpers import load_checkpoint
        load_checkpoint(model, best_ckpt, device=device)

    logger.info("Evaluating on test set...")
    metrics = evaluate_model(model, test_loader, device=device)
    print_metrics_table(metrics, dataset_name=dataset_name.upper())

    # Save metrics
    metrics_save = {k: v for k, v in metrics.items() if k != "confusion_matrix"}
    metrics_path = os.path.join(output_dir, f"metrics_{dataset_name}.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics_save, f, indent=2)
    logger.info(f"Metrics saved to {metrics_path}")

    logger.info("Done.")


if __name__ == "__main__":
    main()
