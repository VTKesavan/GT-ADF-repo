"""
Feature preprocessing pipeline for network intrusion detection datasets.

Steps:
    1. Drop constant / near-zero-variance columns
    2. Handle infinities and NaN values
    3. Drop high-correlation redundant features
    4. One-hot encode categorical features
    5. Min-Max normalization (as used in the paper)
"""

import logging
from typing import Tuple, Optional

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.feature_selection import VarianceThreshold

logger = logging.getLogger(__name__)

# Features known to be uninformative or non-numeric in standard IDS datasets
DROP_COLS = [
    "Flow ID", "Source IP", "Destination IP", "Source Port",
    "Destination Port", "Timestamp", "timestamp", "src_ip",
    "dst_ip", "src_port", "dst_port", "flow_id",
]

# Categorical feature columns in various datasets
CATEGORICAL_COLS = ["proto", "service", "state", "attack_cat", "type"]


class Preprocessor:
    """
    Stateful preprocessing pipeline (fit on training data, transform test data).

    Usage:
        prep = Preprocessor()
        X_train, y_train = prep.fit_transform(df_train, label_col='label')
        X_test,  y_test  = prep.transform(df_test,  label_col='label')
    """

    def __init__(
        self,
        variance_threshold: float = 0.0,
        correlation_threshold: float = 0.98,
        scaler_range: Tuple[float, float] = (0.0, 1.0),
    ):
        self.variance_threshold = variance_threshold
        self.correlation_threshold = correlation_threshold
        self.scaler_range = scaler_range

        self.scaler = MinMaxScaler(feature_range=scaler_range)
        self.var_selector = VarianceThreshold(threshold=variance_threshold)
        self.label_encoders: dict = {}
        self.selected_columns: Optional[list] = None
        self.is_fitted = False

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit_transform(
        self,
        df: pd.DataFrame,
        label_col: str = "label_enc",
        external_labels: Optional[np.ndarray] = None,
    ) -> Tuple[pd.DataFrame, np.ndarray]:
        """
        Fit the preprocessor on df and transform it.

        Args:
            df: Raw feature DataFrame (may contain label column).
            label_col: Name of label column inside df (ignored if external_labels given).
            external_labels: Pre-encoded label array (overrides label_col).

        Returns:
            (feature_df, labels) after full preprocessing.
        """
        # Extract labels
        if external_labels is not None:
            labels = external_labels
            feature_df = df.copy()
        else:
            if label_col in df.columns:
                labels = df[label_col].values
                feature_df = df.drop(columns=[label_col])
            else:
                raise ValueError(f"Label column '{label_col}' not found in DataFrame.")

        feature_df = self._basic_clean(feature_df)
        feature_df = self._encode_categoricals(feature_df, fit=True)
        feature_df = self._select_features(feature_df, fit=True)
        feature_df = self._scale(feature_df, fit=True)

        self.is_fitted = True
        logger.info(f"Preprocessor fitted. Feature shape: {feature_df.shape}")
        return feature_df, labels.astype(np.int64)

    def transform(
        self,
        df: pd.DataFrame,
        label_col: str = "label_enc",
        external_labels: Optional[np.ndarray] = None,
    ) -> Tuple[pd.DataFrame, np.ndarray]:
        """
        Transform new data using a previously fitted preprocessor.
        """
        assert self.is_fitted, "Call fit_transform first."

        if external_labels is not None:
            labels = external_labels
            feature_df = df.copy()
        else:
            if label_col in df.columns:
                labels = df[label_col].values
                feature_df = df.drop(columns=[label_col])
            else:
                raise ValueError(f"Label column '{label_col}' not found.")

        feature_df = self._basic_clean(feature_df)
        feature_df = self._encode_categoricals(feature_df, fit=False)
        feature_df = self._align_columns(feature_df)
        feature_df = self._scale(feature_df, fit=False)

        return feature_df, labels.astype(np.int64)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _basic_clean(self, df: pd.DataFrame) -> pd.DataFrame:
        """Drop uninformative columns, handle NaN/Inf."""
        # Drop known non-feature columns
        cols_to_drop = [c for c in DROP_COLS if c in df.columns]
        df = df.drop(columns=cols_to_drop)

        # Replace inf with NaN, then fill with column median
        df = df.replace([np.inf, -np.inf], np.nan)

        # Convert to numeric where possible
        for col in df.columns:
            if col not in CATEGORICAL_COLS:
                df[col] = pd.to_numeric(df[col], errors="coerce")

        # Fill NaN
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())
        df = df.fillna("unknown")

        return df

    def _encode_categoricals(self, df: pd.DataFrame, fit: bool = True) -> pd.DataFrame:
        """Label-encode categorical columns."""
        cat_cols = [c for c in CATEGORICAL_COLS if c in df.columns]
        for col in cat_cols:
            if fit:
                le = LabelEncoder()
                df[col] = le.fit_transform(df[col].astype(str))
                self.label_encoders[col] = le
            else:
                if col in self.label_encoders:
                    le = self.label_encoders[col]
                    known = set(le.classes_)
                    df[col] = df[col].astype(str).apply(
                        lambda v: v if v in known else le.classes_[0]
                    )
                    df[col] = le.transform(df[col])
        return df

    def _select_features(self, df: pd.DataFrame, fit: bool = True) -> pd.DataFrame:
        """Remove near-zero-variance and high-correlation features."""
        numeric_df = df.select_dtypes(include=[np.number])

        if fit:
            # Variance filter
            self.var_selector.fit(numeric_df)
            mask = self.var_selector.get_support()
            numeric_df = numeric_df.loc[:, mask]

            # Correlation filter
            corr_matrix = numeric_df.corr().abs()
            upper = corr_matrix.where(
                np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
            )
            drop_corr = [
                col for col in upper.columns
                if any(upper[col] > self.correlation_threshold)
            ]
            numeric_df = numeric_df.drop(columns=drop_corr)
            self.selected_columns = list(numeric_df.columns)
            logger.info(f"Selected {len(self.selected_columns)} features after filtering.")
        else:
            numeric_df = numeric_df.reindex(columns=self.selected_columns, fill_value=0.0)

        return numeric_df

    def _align_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Align test dataframe columns to training columns."""
        assert self.selected_columns is not None
        return df.reindex(columns=self.selected_columns, fill_value=0.0)

    def _scale(self, df: pd.DataFrame, fit: bool = True) -> pd.DataFrame:
        """Apply Min-Max normalization."""
        if fit:
            scaled = self.scaler.fit_transform(df)
        else:
            scaled = self.scaler.transform(df)
        return pd.DataFrame(scaled, columns=df.columns, index=df.index)

    @property
    def n_features(self) -> int:
        if self.selected_columns is None:
            raise RuntimeError("Preprocessor not yet fitted.")
        return len(self.selected_columns)
