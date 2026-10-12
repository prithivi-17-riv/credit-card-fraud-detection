"""Model Architecture Module for Imbalanced Fraud Detection.
Defines four distinct models designed specifically to combat high class imbalance:
1. Cost-Sensitive Logistic Regression (class_weight='balanced')
2. SMOTE Over-sampling + Logistic Regression Pipeline
3. Balanced Random Forest Classifier (class_weight='balanced_subsample')
4. XGBoost Classifier with scale_pos_weight adjusted for 0.17% minority prevalence
"""

from typing import Dict, Any
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
import xgboost as xgb


def get_models(imbalance_ratio: float = 577.0, random_state: int = 42) -> Dict[str, Any]:
    """Instantiate a suite of models tailored for imbalanced fraud detection.

    Args:
        imbalance_ratio: Ratio of negative to positive samples (approx 577 for Kaggle dataset)
        random_state: Seed for reproducibility

    Returns:
        Dictionary mapping model names to model instances/pipelines.
    """
    models = {
        "Cost-Sensitive Logistic Regression": LogisticRegression(
            class_weight="balanced",
            max_iter=1000,
            C=0.1,
            solver="lbfgs",
            random_state=random_state,
        ),
        "SMOTE + Logistic Regression": ImbPipeline([
            ("smote", SMOTE(sampling_strategy=0.1, random_state=random_state)),
            ("clf", LogisticRegression(max_iter=1000, C=0.1, random_state=random_state)),
        ]),
        "Balanced Random Forest": RandomForestClassifier(
            n_estimators=100,
            max_depth=12,
            min_samples_split=5,
            class_weight="balanced_subsample",
            n_jobs=-1,
            random_state=random_state,
        ),
        "XGBoost (Cost-Weighted)": xgb.XGBClassifier(
            n_estimators=120,
            max_depth=5,
            learning_rate=0.1,
            scale_pos_weight=imbalance_ratio,
            subsample=0.8,
            colsample_bytree=0.8,
            eval_metric="aucpr",
            random_state=random_state,
            n_jobs=-1,
        ),
    }
    return models
