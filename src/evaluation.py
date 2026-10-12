"""Class-wise Evaluation Module for Credit Card Fraud Detection.
Implements granular class-wise performance evaluation, confusion matrix breakdown,
PR-AUC, ROC-AUC, and decision threshold optimization.
"""

from typing import Dict, Any, List, Tuple
import numpy as np
import pandas as pd
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    roc_auc_score,
    average_precision_score,
    balanced_accuracy_score,
    matthews_corrcoef,
    precision_recall_curve,
    roc_curve,
)


def compute_classwise_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray = None,
) -> Dict[str, Any]:
    """Calculate comprehensive class-wise and global metrics for imbalanced evaluation.

    Args:
        y_true: Ground truth binary labels (0 or 1)
        y_pred: Predicted binary labels (0 or 1)
        y_prob: Predicted probability for positive class (Class 1)

    Returns:
        Dict containing class-wise metrics, confusion matrix, and curve scores.
    """
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()

    # Class 0 (Legitimate Transactions)
    precision_0 = tn / (tn + fn) if (tn + fn) > 0 else 0.0
    recall_0 = tn / (tn + fp) if (tn + fp) > 0 else 0.0  # Specificity
    f1_0 = (2 * precision_0 * recall_0) / (precision_0 + recall_0) if (precision_0 + recall_0) > 0 else 0.0
    support_0 = int(tn + fp)

    # Class 1 (Fraudulent Transactions)
    precision_1 = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall_1 = tp / (tp + fn) if (tp + fn) > 0 else 0.0  # Sensitivity / Detection Rate
    f1_1 = (2 * precision_1 * recall_1) / (precision_1 + recall_1) if (precision_1 + recall_1) > 0 else 0.0
    support_1 = int(tp + fn)

    # Error Rates
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0  # False Positive Rate
    fnr = fn / (fn + tp) if (fn + tp) > 0 else 0.0  # False Negative Rate (Missed Fraud Rate)

    # Global / Imbalanced Metrics
    balanced_acc = balanced_accuracy_score(y_true, y_pred)
    mcc = matthews_corrcoef(y_true, y_pred)

    roc_auc = None
    pr_auc = None
    if y_prob is not None:
        try:
            roc_auc = float(roc_auc_score(y_true, y_prob))
            pr_auc = float(average_precision_score(y_true, y_prob))
        except Exception:
            pass

    return {
        "confusion_matrix": {
            "tn": int(tn),
            "fp": int(fp),
            "fn": int(fn),
            "tp": int(tp),
            "raw_array": cm.tolist(),
        },
        "class_0_legitimate": {
            "precision": float(precision_0),
            "recall_specificity": float(recall_0),
            "f1_score": float(f1_0),
            "support": support_0,
        },
        "class_1_fraud": {
            "precision": float(precision_1),
            "recall_sensitivity": float(recall_1),
            "f1_score": float(f1_1),
            "support": support_1,
            "false_negative_rate": float(fnr),
            "false_positive_rate": float(fpr),
        },
        "overall_summary": {
            "balanced_accuracy": float(balanced_acc),
            "mcc": float(mcc),
            "roc_auc": roc_auc,
            "pr_auc": pr_auc,
            "macro_f1": float((f1_0 + f1_1) / 2.0),
        },
    }


def optimize_threshold(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    thresholds: np.ndarray = None,
) -> Dict[str, Any]:
    """Find the optimal probability threshold that maximizes F1-Score for Class 1 (Fraud).

    In severe class imbalance, the default 0.5 threshold is arbitrary.
    Threshold tuning allows businesses to navigate the precision vs recall frontier.

    Args:
        y_true: Ground truth binary labels
        y_prob: Predicted probabilities of class 1
        thresholds: Array of candidate thresholds to test

    Returns:
        Dictionary with optimal threshold, best F1 score, and performance curves.
    """
    if thresholds is None:
        thresholds = np.linspace(0.01, 0.99, 100)

    precisions = []
    recalls = []
    f1_scores = []

    best_thresh = 0.5
    best_f1 = -1.0
    best_metrics = {}

    for t in thresholds:
        preds = (y_prob >= t).astype(int)
        tp = np.sum((y_true == 1) & (preds == 1))
        fp = np.sum((y_true == 0) & (preds == 1))
        fn = np.sum((y_true == 1) & (preds == 0))

        p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * p * r) / (p + r) if (p + r) > 0 else 0.0

        precisions.append(p)
        recalls.append(r)
        f1_scores.append(f1)

        if f1 > best_f1:
            best_f1 = f1
            best_thresh = t
            best_metrics = {"precision": p, "recall": r, "f1": f1}

    return {
        "optimal_threshold": float(best_thresh),
        "best_fraud_f1": float(best_f1),
        "best_metrics": best_metrics,
        "curve_data": {
            "thresholds": thresholds.tolist(),
            "precisions": precisions,
            "recalls": recalls,
            "f1_scores": f1_scores,
        },
    }


def format_metrics_table(results_dict: Dict[str, Dict[str, Any]]) -> pd.DataFrame:
    """Format model comparison results into a clean tabular DataFrame highlighting class-wise metrics.

    Args:
        results_dict: Dictionary mapping model name to results of compute_classwise_metrics

    Returns:
        pd.DataFrame formatted for presentation
    """
    rows = []
    for model_name, res in results_dict.items():
        c0 = res["class_0_legitimate"]
        c1 = res["class_1_fraud"]
        ov = res["overall_summary"]
        cm = res["confusion_matrix"]

        rows.append({
            "Model": model_name,
            "Legit Precision (C0)": f"{c0['precision']:.4f}",
            "Legit Recall (C0)": f"{c0['recall_specificity']:.4f}",
            "Fraud Precision (C1)": f"{c1['precision']:.4f}",
            "Fraud Recall (C1)": f"{c1['recall_sensitivity']:.4f}",
            "Fraud F1-Score (C1)": f"{c1['f1_score']:.4f}",
            "PR-AUC": f"{ov['pr_auc']:.4f}" if ov["pr_auc"] is not None else "N/A",
            "ROC-AUC": f"{ov['roc_auc']:.4f}" if ov["roc_auc"] is not None else "N/A",
            "Balanced Acc": f"{ov['balanced_accuracy']:.4f}",
            "TP (Caught Fraud)": cm["tp"],
            "FN (Missed Fraud)": cm["fn"],
            "FP (False Alarm)": cm["fp"],
        })
    return pd.DataFrame(rows)
