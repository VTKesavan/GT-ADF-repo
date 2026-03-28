"""
Training loop for GT-ADF with semi-supervised hybrid loss.

"""

import os
import time
import logging
from typing import Optional, Dict, Any

import torch
import torch.nn as nn
import torch.optim as optim
from torch_geometric.loader import DataLoader
from torch.utils.tensorboard import SummaryWriter

from ..models.gt_adf import GTADF, HybridLoss
from ..evaluation.metrics import compute_metrics
from ..utils.helpers import set_seed, save_checkpoint, load_checkpoint

logger = logging.getLogger(__name__)


class Trainer:
    """
    GT-ADF Trainer.

    Args:
        model (GTADF): The GT-ADF model.
        config (dict): Training configuration dictionary.
        device (str): 'cuda' or 'cpu'.
        output_dir (str): Directory to save checkpoints and logs.
    """

    def __init__(
        self,
        model: GTADF,
        config: Dict[str, Any],
        device: str = "cpu",
        output_dir: str = "results",
    ):
        self.model = model.to(device)
        self.config = config
        self.device = device
        self.output_dir = output_dir

        os.makedirs(os.path.join(output_dir, "checkpoints"), exist_ok=True)
        os.makedirs(os.path.join(output_dir, "logs"), exist_ok=True)

        # Optimizer
        self.optimizer = optim.Adam(
            model.parameters(),
            lr=config.get("lr", 0.001),
            weight_decay=config.get("weight_decay", 1e-5),
        )

        # Learning rate scheduler
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode="min",
            factor=0.5,
            patience=5,
        )

        # Loss function
        self.criterion = HybridLoss(
            lambda_unsup=config.get("lambda_unsup", 0.5),
            num_classes=config.get("num_classes", 2),
        )

        # TensorBoard
        log_dir = os.path.join(output_dir, "logs")
        self.writer = SummaryWriter(log_dir=log_dir)

        # Training state
        self.best_val_loss = float("inf")
        self.patience_counter = 0
        self.global_step = 0

    def train(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        max_epochs: int = 100,
        patience: int = 10,
        labeled_ratio: float = 0.2,
    ) -> Dict[str, Any]:
        """
        Main training loop with early stopping.

        Args:
            train_loader: DataLoader for training graphs.
            val_loader: DataLoader for validation graphs.
            max_epochs: Maximum number of training epochs.
            patience: Early stopping patience.
            labeled_ratio: Fraction of nodes treated as labeled (semi-supervised).

        Returns:
            History dictionary with train/val losses and metrics.
        """
        history = {
            "train_loss": [],
            "val_loss": [],
            "val_accuracy": [],
            "val_f1": [],
        }

        logger.info(
            f"Starting training: max_epochs={max_epochs}, patience={patience}, "
            f"labeled_ratio={labeled_ratio:.0%}, device={self.device}"
        )

        for epoch in range(1, max_epochs + 1):
            t0 = time.time()

            # Training phase
            train_loss = self._train_epoch(train_loader, labeled_ratio)

            # Validation phase
            val_loss, val_metrics = self._val_epoch(val_loader)

            # Scheduler step
            self.scheduler.step(val_loss)

            # Log
            elapsed = time.time() - t0
            logger.info(
                f"Epoch {epoch:03d}/{max_epochs} | "
                f"Train Loss: {train_loss:.4f} | "
                f"Val Loss: {val_loss:.4f} | "
                f"Val Acc: {val_metrics['accuracy']:.4f} | "
                f"Val F1: {val_metrics['f1_macro']:.4f} | "
                f"Time: {elapsed:.1f}s"
            )

            # TensorBoard
            self.writer.add_scalar("Loss/train", train_loss, epoch)
            self.writer.add_scalar("Loss/val", val_loss, epoch)
            self.writer.add_scalar("Accuracy/val", val_metrics["accuracy"], epoch)
            self.writer.add_scalar("F1/val", val_metrics["f1_macro"], epoch)

            history["train_loss"].append(train_loss)
            history["val_loss"].append(val_loss)
            history["val_accuracy"].append(val_metrics["accuracy"])
            history["val_f1"].append(val_metrics["f1_macro"])

            # Checkpoint saving
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.patience_counter = 0
                ckpt_path = os.path.join(
                    self.output_dir, "checkpoints", "gt_adf_best.pt"
                )
                save_checkpoint(self.model, self.optimizer, epoch, val_loss, ckpt_path)
                logger.info(f"  ✓ Best model saved to {ckpt_path}")
            else:
                self.patience_counter += 1
                if self.patience_counter >= patience:
                    logger.info(
                        f"Early stopping triggered at epoch {epoch} "
                        f"(patience={patience})."
                    )
                    break

        self.writer.close()
        logger.info(f"Training complete. Best val loss: {self.best_val_loss:.4f}")
        return history

    def _train_epoch(self, loader: DataLoader, labeled_ratio: float) -> float:
        """Single training epoch."""
        self.model.train()
        total_loss = 0.0

        for batch in loader:
            batch = batch.to(self.device)
            self.optimizer.zero_grad()

            # Forward pass
            logits, node_embeddings, anomaly_scores = self.model(
                batch.x, batch.edge_index, batch.batch
            )

            # Semi-supervised split: labeled vs unlabeled nodes
            n_nodes = batch.x.size(0)
            n_labeled = max(1, int(n_nodes * labeled_ratio))
            labeled_idx = torch.randperm(n_nodes, device=self.device)[:n_labeled]
            unlabeled_idx = torch.randperm(n_nodes, device=self.device)[n_labeled:]

            # Node-level labels (from node_labels attribute)
            if hasattr(batch, "node_labels") and batch.node_labels is not None:
                node_labels = batch.node_labels.to(self.device)
            else:
                # Fall back to graph-level label replicated to all nodes
                node_labels = batch.y[batch.batch]

            # Supervised loss (graph level using logits)
            l_sup = nn.CrossEntropyLoss()(logits, batch.y.to(self.device))

            # Unsupervised consistency loss via feature perturbation
            if len(unlabeled_idx) > 0:
                x_perturbed = batch.x.clone()
                noise = torch.randn_like(x_perturbed) * 0.01
                x_perturbed = x_perturbed + noise

                logits_perturbed, _, _ = self.model(
                    x_perturbed, batch.edge_index, batch.batch
                )

                l_unsup = nn.KLDivLoss(reduction="batchmean")(
                    nn.functional.log_softmax(logits_perturbed, dim=-1),
                    nn.functional.softmax(logits, dim=-1).detach(),
                )
            else:
                l_unsup = torch.tensor(0.0, device=self.device)

            loss = l_sup + self.model.lambda_unsup * l_unsup
            loss.backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

            self.optimizer.step()
            total_loss += loss.item()
            self.global_step += 1

        return total_loss / max(1, len(loader))

    @torch.no_grad()
    def _val_epoch(self, loader: DataLoader):
        """Single validation epoch."""
        self.model.eval()
        total_loss = 0.0
        all_preds, all_labels = [], []

        for batch in loader:
            batch = batch.to(self.device)
            logits, _, _ = self.model(batch.x, batch.edge_index, batch.batch)

            loss = nn.CrossEntropyLoss()(logits, batch.y.to(self.device))
            total_loss += loss.item()

            preds = logits.argmax(dim=-1).cpu().numpy()
            labels = batch.y.cpu().numpy()
            all_preds.extend(preds.tolist())
            all_labels.extend(labels.tolist())

        avg_loss = total_loss / max(1, len(loader))
        metrics = compute_metrics(all_labels, all_preds)
        return avg_loss, metrics
