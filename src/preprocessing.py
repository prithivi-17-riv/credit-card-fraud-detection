"""Preprocessing Module for Credit Card Fraud Detection.
Implements robust feature scaling adhering strictly to featurization ordering
(fit strictly on training data, transform both train and test).
"""

from typing import Tuple, List
import pandas as pd
import numpy as np
from sklearn.preprocessing import RobustScaler
import joblib


class FraudDataPreprocessor:
    """Preprocessor for Credit Card Fraud Detection.
    Uses RobustScaler for 'Amount' and 'Time' to handle extreme outliers,
    while retaining the PCA-engineered properties of features V1 to V28.
    """

    def __init__(self, cols_to_scale: List[str] = None):
        if cols_to_scale is None:
            self.cols_to_scale = ["Time", "Amount"]
        else:
            self.cols_to_scale = cols_to_scale
        self.scaler = RobustScaler()
        self.is_fitted = False

    def fit(self, X_train: pd.DataFrame) -> "FraudDataPreprocessor":
        """Fit scaler on training data only.

        Args:
            X_train: Training features DataFrame

        Returns:
            self
        """
        self.feature_names_ = list(X_train.columns)
        self.scaler.fit(X_train[self.cols_to_scale])
        self.is_fitted = True
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Transform features using fitted scaler.

        Args:
            X: Input features DataFrame

        Returns:
            Transformed features DataFrame with preserved column ordering
        """
        if not self.is_fitted:
            raise RuntimeError("Preprocessor must be fitted on training data before calling transform.")

        # Reorder columns to match training order if all present
        if hasattr(self, "feature_names_") and self.feature_names_:
            missing = set(self.feature_names_) - set(X.columns)
            if not missing:
                X = X[self.feature_names_]

        X_transformed = X.copy()
        scaled_values = self.scaler.transform(X[self.cols_to_scale])
        X_transformed[self.cols_to_scale] = scaled_values
        return X_transformed

    def fit_transform(self, X_train: pd.DataFrame) -> pd.DataFrame:
        """Fit on training data and return transformed DataFrame.

        Args:
            X_train: Training features DataFrame

        Returns:
            Transformed training features
        """
        return self.fit(X_train).transform(X_train)

    def save(self, file_path: str) -> None:
        """Save fitted preprocessor to disk."""
        joblib.dump(self, file_path)

    @staticmethod
    def load(file_path: str) -> "FraudDataPreprocessor":
        """Load fitted preprocessor from disk."""
        return joblib.load(file_path)


def preprocess_features(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame, FraudDataPreprocessor]:
    """Preprocess train and test features ensuring strict ordering.

    Args:
        X_train: Training features
        X_test: Testing features

    Returns:
        X_train_scaled, X_test_scaled, fitted_preprocessor
    """
    preprocessor = FraudDataPreprocessor()
    X_train_scaled = preprocessor.fit_transform(X_train)
    X_test_scaled = preprocessor.transform(X_test)
    return X_train_scaled, X_test_scaled, preprocessor
