"""Data Loader Module for Credit Card Fraud Detection.
Handles dataset loading, validation, summary statistics, and stratified train/test splitting.
"""

from typing import Tuple, Dict, Any
import os
import pandas as pd
from sklearn.model_selection import train_test_split


def load_dataset(file_path: str) -> pd.DataFrame:
    """Load credit card transactions dataset from CSV.

    Args:
        file_path: Path to creditcard.csv

    Returns:
        pd.DataFrame containing the dataset
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Dataset file not found at: {file_path}")

    df = pd.read_csv(file_path)
    return df


def validate_dataset(df: pd.DataFrame) -> Dict[str, Any]:
    """Validate schema and calculate key statistics.

    Args:
        df: Input DataFrame

    Returns:
        Dictionary containing summary stats and verification flags.
    """
    expected_cols = {"Time", "Amount", "Class"}
    v_cols = {f"V{i}" for i in range(1, 29)}
    expected_all = expected_cols.union(v_cols)

    actual_cols = set(df.columns)
    missing_cols = expected_all - actual_cols

    null_counts = df.isnull().sum()
    total_nulls = int(null_counts.sum())

    total_records = len(df)
    class_counts = df["Class"].value_counts().to_dict()
    fraud_count = int(class_counts.get(1, 0))
    legit_count = int(class_counts.get(0, 0))
    fraud_percentage = (fraud_count / total_records) * 100.0 if total_records > 0 else 0.0

    stats = {
        "total_records": total_records,
        "total_features": len(df.columns) - 1,
        "missing_columns": list(missing_cols),
        "total_nulls": total_nulls,
        "class_0_legit": legit_count,
        "class_1_fraud": fraud_count,
        "fraud_percentage": fraud_percentage,
        "imbalance_ratio": f"1 : {legit_count / fraud_count:.1f}" if fraud_count > 0 else "N/A",
    }
    return stats


def split_data(
    df: pd.DataFrame,
    test_size: float = 0.2,
    random_state: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Perform stratified split into training and test sets.

    Stratification is essential due to the 0.17% minority class to preserve
    identical class proportions in train and test splits.

    Args:
        df: Input DataFrame
        test_size: Proportion of dataset to include in test split (default 0.20)
        random_state: Random seed for reproducibility

    Returns:
        X_train, X_test, y_train, y_test
    """
    X = df.drop(columns=["Class"])
    y = df["Class"].astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        stratify=y,
        random_state=random_state,
    )
    return X_train, X_test, y_train, y_test
