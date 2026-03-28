#!/usr/bin/env python3
"""
Usage:
    python scripts/preprocess.py --dataset cicids2017 \
        --data_dir data/raw/CICIDS2017 --output_dir data/processed
"""

import sys
import os
import argparse
import logging

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data.dataset import load_dataset
from src.utils.helpers import setup_logging

logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="Preprocess IDS dataset into graph format")
    parser.add_argument(
        "--dataset", type=str, required=True,
        choices=["cicids2017", "unsw_nb15", "ton_iot"],
        help="Dataset to preprocess"
    )
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="data/processed")
    parser.add_argument("--binary", action="store_true", help="Use binary labels")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main():
    args = parse_args()
    setup_logging()
    os.makedirs(args.output_dir, exist_ok=True)

    for split in ["train", "test"]:
        logger.info(f"Processing {args.dataset.upper()} [{split}]...")
        dataset = load_dataset(
            dataset_name=args.dataset,
            data_dir=args.data_dir,
            split=split,
            binary=args.binary,
            seed=args.seed,
        )
        logger.info(f"  → {len(dataset)} graphs, feature dim = {dataset[0].x.shape[-1]}")

    logger.info("Preprocessing complete.")


if __name__ == "__main__":
    main()
