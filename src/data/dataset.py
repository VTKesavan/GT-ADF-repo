"""
Dataset loaders for CICIDS2017-SafeML, UNSW-NB15, and ToN-IoT.

Each loader:
    1. Reads the raw CSV files
    2. Applies feature selection and encoding
    3. Constructs PyTorch Geometric Data objects (graph per sample)
    4. Returns train/test splits
"""

import os
import logging
from typing import Tuple, Optional, List

import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data, Dataset, InMemoryDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from .preprocessor import Preprocessor
from .graph_builder import GraphBuilder

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Label maps
# ---------------------------------------------------------------------------

CICIDS2017_LABELS = {
    "BENIGN": 0,
    "Bot": 1,
    "DDoS": 2,
    "DoS GoldenEye": 3,
    "DoS Hulk": 4,
    "DoS Slowhttptest": 5,
    "DoS slowloris": 6,
    "FTP-Patator": 7,
    "Heartbleed": 8,
    "Infiltration": 9,
    "PortScan": 10,
    "SSH-Patator": 11,
    "Web Attack – Brute Force": 12,
    "Web Attack – Sql Injection": 13,
    "Web Attack – XSS": 14,
}

UNSW_NB15_LABELS = {
    "Normal": 0,
    "Fuzzers": 1,
    "Analysis": 2,
    "Backdoors": 3,
    "DoS": 4,
    "Exploits": 5,
    "Generic": 6,
    "Reconnaissance": 7,
    "Shellcode": 8,
    "Worms": 9,
}

TON_IOT_LABELS = {
    "normal": 0,
    "backdoor": 1,
    "ddos": 2,
    "dos": 3,
    "injection": 4,
    "mitm": 5,
    "password": 6,
    "ransomware": 7,
    "scanning": 8,
    "xss": 9,
}


# ---------------------------------------------------------------------------
# CICIDS2017 Dataset
# ---------------------------------------------------------------------------

class CICIDS2017Dataset(InMemoryDataset):
    """
    CICIDS2017-SafeML dataset loader.

    Expected directory structure:
        data_dir/
            Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv
            Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv
            Friday-WorkingHours-Morning.pcap_ISCX.csv
            Monday-WorkingHours.pcap_ISCX.csv
            Thursday-WorkingHours-Afternoon-Infilteration.pcap_ISCX.csv
            Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv
            Tuesday-WorkingHours.pcap_ISCX.csv
            Wednesday-workingHours.pcap_ISCX.csv

    Args:
        data_dir (str): Path to directory containing raw CSVs.
        split (str): 'train' or 'test'.
        test_size (float): Fraction of data for test set.
        binary (bool): If True, collapse all attacks to class 1 (binary classification).
        seed (int): Random seed for reproducibility.
    """

    LABEL_COL = " Label"

    def __init__(
        self,
        data_dir: str,
        split: str = "train",
        test_size: float = 0.2,
        binary: bool = False,
        seed: int = 42,
        transform=None,
        pre_transform=None,
    ):
        self.data_dir = data_dir
        self.split = split
        self.test_size = test_size
        self.binary = binary
        self.seed = seed

        self.preprocessor = Preprocessor()
        self.graph_builder = GraphBuilder()
        self.label_map = CICIDS2017_LABELS

        super().__init__(root=data_dir, transform=transform, pre_transform=pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self) -> List[str]:
        return [f for f in os.listdir(self.data_dir) if f.endswith(".csv")]

    @property
    def processed_file_names(self) -> List[str]:
        return [f"cicids2017_{self.split}.pt"]

    def download(self):
        logger.info(
            "CICIDS2017 dataset not found. Please download from:\n"
            "http://205.174.165.80/CICDataset/CIC-IDS-2017/Dataset/MachineLearningCSV.zip\n"
            f"and place CSV files in: {self.data_dir}"
        )

    def process(self):
        dfs = []
        for fname in self.raw_file_names:
            fpath = os.path.join(self.data_dir, fname)
            try:
                df = pd.read_csv(fpath, low_memory=False, encoding="utf-8")
                dfs.append(df)
                logger.info(f"Loaded {fname}: {len(df)} rows")
            except Exception as e:
                logger.warning(f"Could not load {fname}: {e}")

        if not dfs:
            raise FileNotFoundError(
                f"No CSV files found in {self.data_dir}. "
                "Please download the CICIDS2017 dataset."
            )

        df = pd.concat(dfs, ignore_index=True)
        df.columns = df.columns.str.strip()

        # Map labels
        label_col = self.LABEL_COL.strip()
        df[label_col] = df[label_col].str.strip()
        df["label_enc"] = df[label_col].map(self.label_map).fillna(-1).astype(int)
        df = df[df["label_enc"] >= 0]  # drop unknown labels

        if self.binary:
            df["label_enc"] = (df["label_enc"] > 0).astype(int)

        # Preprocess features
        feature_df, labels = self.preprocessor.fit_transform(df, label_col="label_enc")

        # Train / test split
        X_train, X_test, y_train, y_test = train_test_split(
            feature_df.values,
            labels,
            test_size=self.test_size,
            stratify=labels,
            random_state=self.seed,
        )

        X = X_train if self.split == "train" else X_test
        y = y_train if self.split == "train" else y_test

        # Build graph objects
        data_list = self.graph_builder.build_graphs(X, y)
        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])


# ---------------------------------------------------------------------------
# UNSW-NB15 Dataset
# ---------------------------------------------------------------------------

class UNSWNB15Dataset(InMemoryDataset):
    """
    UNSW-NB15 dataset loader.

    Expected directory structure:
        data_dir/
            UNSW_NB15_training-set.csv
            UNSW_NB15_testing-set.csv
        OR
            UNSW-NB15_1.csv, UNSW-NB15_2.csv, UNSW-NB15_3.csv, UNSW-NB15_4.csv

    Args:
        data_dir (str): Path to directory containing UNSW-NB15 CSVs.
        split (str): 'train' or 'test'.
        binary (bool): If True, use binary classification (normal vs attack).
        seed (int): Random seed.
    """

    LABEL_COL = "label"
    ATTACK_CAT_COL = "attack_cat"

    def __init__(
        self,
        data_dir: str,
        split: str = "train",
        test_size: float = 0.2,
        binary: bool = False,
        seed: int = 42,
        transform=None,
        pre_transform=None,
    ):
        self.data_dir = data_dir
        self.split = split
        self.test_size = test_size
        self.binary = binary
        self.seed = seed

        self.preprocessor = Preprocessor()
        self.graph_builder = GraphBuilder()

        super().__init__(root=data_dir, transform=transform, pre_transform=pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self) -> List[str]:
        return [f for f in os.listdir(self.data_dir) if f.endswith(".csv")]

    @property
    def processed_file_names(self) -> List[str]:
        return [f"unsw_nb15_{self.split}.pt"]

    def download(self):
        logger.info(
            "UNSW-NB15 dataset not found. Please download from:\n"
            "https://research.unsw.edu.au/projects/unsw-nb15-dataset\n"
            f"and place CSV files in: {self.data_dir}"
        )

    def process(self):
        # Try loading pre-split files first
        train_file = os.path.join(self.data_dir, "UNSW_NB15_training-set.csv")
        test_file = os.path.join(self.data_dir, "UNSW_NB15_testing-set.csv")

        if os.path.exists(train_file) and os.path.exists(test_file):
            df = pd.read_csv(train_file if self.split == "train" else test_file)
        else:
            # Load and merge all CSVs
            dfs = []
            for fname in self.raw_file_names:
                df_i = pd.read_csv(os.path.join(self.data_dir, fname), low_memory=False)
                dfs.append(df_i)
            df = pd.concat(dfs, ignore_index=True)

            # Manual split
            X_train, X_test = train_test_split(
                df, test_size=self.test_size, random_state=self.seed
            )
            df = X_train if self.split == "train" else X_test

        df.columns = df.columns.str.strip().str.lower()

        # Encode attack categories as multi-class labels
        le = LabelEncoder()
        if self.ATTACK_CAT_COL in df.columns and not self.binary:
            df[self.ATTACK_CAT_COL] = df[self.ATTACK_CAT_COL].fillna("Normal").str.strip()
            labels = le.fit_transform(df[self.ATTACK_CAT_COL])
        else:
            labels = df[self.LABEL_COL].values.astype(int)

        feature_df, labels = self.preprocessor.fit_transform(
            df.drop(columns=[self.LABEL_COL, self.ATTACK_CAT_COL], errors="ignore"),
            external_labels=labels,
        )

        data_list = self.graph_builder.build_graphs(feature_df.values, labels)
        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])


# ---------------------------------------------------------------------------
# ToN-IoT Dataset
# ---------------------------------------------------------------------------

class ToNIoTDataset(InMemoryDataset):
    """
    ToN-IoT Network dataset loader.

    Expected directory structure:
        data_dir/
            NF-ToN-IoT.csv   OR   Train_Test_Network.csv   OR   *.csv

    Args:
        data_dir (str): Path to directory containing ToN-IoT CSVs.
        split (str): 'train' or 'test'.
        binary (bool): If True, binary classification.
        seed (int): Random seed.
    """

    LABEL_COL = "label"
    TYPE_COL = "type"

    def __init__(
        self,
        data_dir: str,
        split: str = "train",
        test_size: float = 0.2,
        binary: bool = False,
        seed: int = 42,
        transform=None,
        pre_transform=None,
    ):
        self.data_dir = data_dir
        self.split = split
        self.test_size = test_size
        self.binary = binary
        self.seed = seed

        self.preprocessor = Preprocessor()
        self.graph_builder = GraphBuilder()
        self.label_map = TON_IOT_LABELS

        super().__init__(root=data_dir, transform=transform, pre_transform=pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self) -> List[str]:
        return [f for f in os.listdir(self.data_dir) if f.endswith(".csv")]

    @property
    def processed_file_names(self) -> List[str]:
        return [f"ton_iot_{self.split}.pt"]

    def download(self):
        logger.info(
            "ToN-IoT dataset not found. Please download from:\n"
            "https://research.unsw.edu.au/projects/toniot-datasets\n"
            f"and place CSV files in: {self.data_dir}"
        )

    def process(self):
        dfs = []
        for fname in self.raw_file_names:
            df_i = pd.read_csv(
                os.path.join(self.data_dir, fname), low_memory=False
            )
            dfs.append(df_i)

        if not dfs:
            raise FileNotFoundError(
                f"No CSV files found in {self.data_dir}. "
                "Please download the ToN-IoT dataset."
            )

        df = pd.concat(dfs, ignore_index=True)
        df.columns = df.columns.str.strip().str.lower()

        # Encode labels
        if self.TYPE_COL in df.columns and not self.binary:
            df[self.TYPE_COL] = df[self.TYPE_COL].str.strip().str.lower()
            labels = df[self.TYPE_COL].map(self.label_map).fillna(0).astype(int).values
        else:
            labels = df[self.LABEL_COL].values.astype(int)

        drop_cols = [c for c in [self.LABEL_COL, self.TYPE_COL] if c in df.columns]
        feature_df, labels = self.preprocessor.fit_transform(
            df.drop(columns=drop_cols, errors="ignore"),
            external_labels=labels,
        )

        # Split
        X_train, X_test, y_train, y_test = train_test_split(
            feature_df.values,
            labels,
            test_size=self.test_size,
            random_state=self.seed,
        )

        X = X_train if self.split == "train" else X_test
        y = y_train if self.split == "train" else y_test

        data_list = self.graph_builder.build_graphs(X, y)
        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])


# ---------------------------------------------------------------------------
# Factory function
# ---------------------------------------------------------------------------

def load_dataset(
    dataset_name: str,
    data_dir: str,
    split: str = "train",
    binary: bool = False,
    seed: int = 42,
) -> InMemoryDataset:
    """
    Load a dataset by name.

    Args:
        dataset_name: One of 'cicids2017', 'unsw_nb15', 'ton_iot'.
        data_dir: Path to raw data directory.
        split: 'train' or 'test'.
        binary: Binary or multi-class classification.
        seed: Random seed.

    Returns:
        PyTorch Geometric InMemoryDataset.
    """
    registry = {
        "cicids2017": CICIDS2017Dataset,
        "unsw_nb15": UNSWNB15Dataset,
        "ton_iot": ToNIoTDataset,
    }

    name_lower = dataset_name.lower().replace("-", "_")
    if name_lower not in registry:
        raise ValueError(
            f"Unknown dataset '{dataset_name}'. "
            f"Choose from: {list(registry.keys())}"
        )

    cls = registry[name_lower]
    return cls(data_dir=data_dir, split=split, binary=binary, seed=seed)
