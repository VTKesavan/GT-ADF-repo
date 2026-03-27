"""
Utility helpers: seeding, checkpointing, logging setup.
"""

import os
import random
import logging
from typing import Optional

import numpy as np
import torch


def set_seed(seed: int = 42) -> None:
    """Set random seeds for full reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def get_device(prefer_gpu: bool = True) -> str:
    """Return 'cuda' if available and preferred, else 'cpu'."""
    if prefer_gpu and torch.cuda.is_available():
        device = "cuda"
        gpu_name = torch.cuda.get_device_name(0)
        logging.getLogger(__name__).info(f"Using GPU: {gpu_name}")
    else:
        device = "cpu"
        logging.getLogger(__name__).info("Using CPU.")
    return device


def setup_logging(level: str = "INFO", log_file: Optional[str] = None) -> None:
    """Configure root logger."""
    handlers = [logging.StreamHandler()]
    if log_file:
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        handlers.append(logging.FileHandler(log_file))

    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=handlers,
    )


def save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    val_loss: float,
    path: str,
) -> None:
    """Save a model checkpoint."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "val_loss": val_loss,
        },
        path,
    )


def load_checkpoint(
    model: torch.nn.Module,
    path: str,
    optimizer: Optional[torch.optim.Optimizer] = None,
    device: str = "cpu",
) -> dict:
    """Load a checkpoint into model (and optionally optimizer)."""
    checkpoint = torch.load(path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    logging.getLogger(__name__).info(
        f"Loaded checkpoint from {path} (epoch {checkpoint['epoch']}, "
        f"val_loss={checkpoint['val_loss']:.4f})"
    )
    return checkpoint


def count_parameters(model: torch.nn.Module) -> int:
    """Count trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def model_summary(model: torch.nn.Module) -> None:
    """Print a brief model summary."""
    n_params = count_parameters(model)
    print(f"\n{'='*50}")
    print(f"  GT-ADF Model Summary")
    print(f"{'='*50}")
    print(model)
    print(f"\n  Trainable parameters: {n_params:,} ({n_params/1e6:.2f}M)")
    print(f"{'='*50}\n")
