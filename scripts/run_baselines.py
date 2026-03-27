#!/usr/bin/env python3
"""
Train and evaluate all baseline models (GNN, CNN, LSTM, SVM, RF)
against GT-ADF on a given dataset.

Usage:
    python scripts/run_baselines.py --config configs/cicids2017.yaml
"""

import sys
import os
import argparse
import json
import logging
import time

import yaml
import numpy as np
import torch
from torch_geometric.loader import DataLoader

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data.dataset import load_dataset
from src.models.baselines import GNNBaseline, CNNBaseline, LSTMBaseline, SVMBaseline, RandomForestBaseline
from src.models.gt_adf import GTADF
from src.evaluation.metrics import evaluate_model, compute_metrics, print_metrics_table
from src.utils.helpers import set_seed, get_device, setup_logging, load_checkpoint

logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="Run all baselines + GT-ADF")
    parser.add_argument("--config", type=str, required=True)
    return parser.parse_args()


def train_torch_model(model, train_loader, val_loader, device, epochs=50, patience=10):
    """Generic training loop for PyTorch models (GNN, CNN, LSTM)."""
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = torch.nn.CrossEntropyLoss()
    best_val_loss = float("inf")
    patience_counter = 0

    model.to(device)
    for epoch in range(1, epochs + 1):
        model.train()
        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            logits = model(batch.x, batch.edge_index, batch.batch)
            loss = criterion(logits, batch.y)
            loss.backward()
            optimizer.step()

        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(device)
                logits = model(batch.x, batch.edge_index, batch.batch)
                val_loss += criterion(logits, batch.y).item()
        val_loss /= max(1, len(val_loader))

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                logger.info(f"  Early stop at epoch {epoch}")
                break

    return model


def eval_torch_model(model, loader, device):
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            logits = model(batch.x, batch.edge_index, batch.batch)
            preds = logits.argmax(dim=-1).cpu().numpy()
            all_preds.extend(preds.tolist())
            all_labels.extend(batch.y.cpu().numpy().tolist())
    return compute_metrics(all_labels, all_preds)


def graphs_to_arrays(loader):
    """Extract flat feature arrays from graph loader for sklearn models."""
    X_list, y_list = [], []
    for batch in loader:
        X_list.append(batch.x.numpy())
        # replicate graph label to all nodes
        node_labels = batch.y[batch.batch].numpy()
        y_list.append(node_labels)
    return np.concatenate(X_list), np.concatenate(y_list)


def main():
    args = parse_args()
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    setup_logging()
    set_seed(config.get("seed", 42))
    device = get_device(prefer_gpu=config.get("use_gpu", True))

    dataset_name = config["dataset"]
    data_dir = config["data_dir"]
    num_classes = config.get("num_classes", 2)

    train_dataset = load_dataset(dataset_name, data_dir, "train",
                                  binary=config.get("binary", False))
    test_dataset = load_dataset(dataset_name, data_dir, "test",
                                 binary=config.get("binary", False))

    val_size = max(1, int(len(train_dataset) * 0.2))
    train_size = len(train_dataset) - val_size
    train_sub, val_sub = torch.utils.data.random_split(
        train_dataset, [train_size, val_size]
    )

    batch_size = config.get("batch_size", 64)
    train_loader = DataLoader(train_sub, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_sub, batch_size=batch_size)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)

    sample = train_dataset[0]
    in_channels = sample.x.size(-1)

    all_results = {}

    # ----------------------------------------------------------------
    # GT-ADF (proposed)
    # ----------------------------------------------------------------
    logger.info("=== GT-ADF (Proposed) ===")
    from src.training.trainer import Trainer
    gt_adf = GTADF(in_channels, hidden_channels=128, out_channels=num_classes,
                   num_layers=4, num_heads=8, dropout=0.2)
    trainer = Trainer(gt_adf, config, device=device, output_dir=config.get("output_dir", "results"))
    t0 = time.time()
    trainer.train(train_loader, val_loader, max_epochs=config.get("max_epochs", 100),
                  patience=config.get("patience", 10))
    train_time = time.time() - t0

    best_ckpt = os.path.join(config.get("output_dir", "results"), "checkpoints", "gt_adf_best.pt")
    if os.path.exists(best_ckpt):
        load_checkpoint(gt_adf, best_ckpt, device=device)
    t_inf = time.time()
    metrics = evaluate_model(gt_adf, test_loader, device=device)
    metrics["inference_time_ms"] = (time.time() - t_inf) / len(test_dataset) * 1000
    all_results["GT-ADF (Proposed)"] = metrics
    print_metrics_table(metrics, "GT-ADF")

    # ----------------------------------------------------------------
    # GNN Baseline
    # ----------------------------------------------------------------
    logger.info("=== GNN Baseline ===")
    gnn = GNNBaseline(in_channels, hidden_channels=128, out_channels=num_classes)
    gnn = train_torch_model(gnn, train_loader, val_loader, device, epochs=50)
    t_inf = time.time()
    metrics = eval_torch_model(gnn, test_loader, device)
    metrics["inference_time_ms"] = (time.time() - t_inf) / len(test_dataset) * 1000
    all_results["GNN"] = metrics
    print_metrics_table(metrics, "GNN")

    # ----------------------------------------------------------------
    # CNN-IDS Baseline
    # ----------------------------------------------------------------
    logger.info("=== CNN-IDS Baseline ===")
    cnn = CNNBaseline(in_channels, hidden_channels=128, out_channels=num_classes)
    cnn = train_torch_model(cnn, train_loader, val_loader, device, epochs=50)
    t_inf = time.time()
    metrics = eval_torch_model(cnn, test_loader, device)
    metrics["inference_time_ms"] = (time.time() - t_inf) / len(test_dataset) * 1000
    all_results["CNN-IDS"] = metrics
    print_metrics_table(metrics, "CNN-IDS")

    # ----------------------------------------------------------------
    # LSTM Baseline
    # ----------------------------------------------------------------
    logger.info("=== LSTM Baseline ===")
    lstm = LSTMBaseline(in_channels, hidden_channels=128, out_channels=num_classes)
    lstm = train_torch_model(lstm, train_loader, val_loader, device, epochs=50)
    t_inf = time.time()
    metrics = eval_torch_model(lstm, test_loader, device)
    metrics["inference_time_ms"] = (time.time() - t_inf) / len(test_dataset) * 1000
    all_results["LSTM"] = metrics
    print_metrics_table(metrics, "LSTM")

    # ----------------------------------------------------------------
    # SVM and Random Forest (sklearn — need flat arrays)
    # ----------------------------------------------------------------
    logger.info("Extracting flat arrays for sklearn baselines...")
    X_train_flat, y_train_flat = graphs_to_arrays(train_loader)
    X_test_flat, y_test_flat = graphs_to_arrays(test_loader)

    for name, cls in [("SVM", SVMBaseline), ("Random Forest", RandomForestBaseline)]:
        logger.info(f"=== {name} Baseline ===")
        baseline = cls()
        baseline.fit(X_train_flat, y_train_flat)
        t_inf = time.time()
        preds = baseline.predict(X_test_flat)
        inf_time = (time.time() - t_inf) / len(X_test_flat) * 1000
        metrics = compute_metrics(y_test_flat.tolist(), preds.tolist())
        metrics["inference_time_ms"] = inf_time
        all_results[name] = metrics
        print_metrics_table(metrics, name)

    # ----------------------------------------------------------------
    # Save comparison table
    # ----------------------------------------------------------------
    output_dir = config.get("output_dir", "results")
    os.makedirs(output_dir, exist_ok=True)
    comparison_path = os.path.join(output_dir, f"baseline_comparison_{dataset_name}.json")
    save_results = {}
    for model_name, m in all_results.items():
        save_results[model_name] = {
            k: v for k, v in m.items() if k != "confusion_matrix"
        }
    with open(comparison_path, "w") as f:
        json.dump(save_results, f, indent=2)
    logger.info(f"\nAll results saved to {comparison_path}")

    # Summary table
    print(f"\n{'='*80}")
    print(f"{'Model':<25} {'Accuracy':>10} {'F1':>10} {'AUC':>10} {'Latency(ms)':>14}")
    print(f"{'-'*80}")
    for model_name, m in all_results.items():
        acc = m.get("accuracy", 0.0) * 100
        f1 = m.get("f1_macro", 0.0) * 100
        auc = m.get("roc_auc", 0.0) * 100 if m.get("roc_auc") else 0.0
        lat = m.get("inference_time_ms", 0.0)
        marker = " ◀" if "GT-ADF" in model_name else ""
        print(f"{model_name:<25} {acc:>9.1f}% {f1:>9.1f}% {auc:>9.1f}% {lat:>12.2f}ms{marker}")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    main()
