"""Visualization Module for Credit Card Fraud Detection.
Produces high-resolution, publication-ready figures for EDA, class-wise metrics,
confusion matrices, PR/ROC curves, and threshold optimization.
"""

from typing import Dict, Any, List
import os
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_curve, roc_curve, auc

# Clean visual style
plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
plt.rcParams["font.sans-serif"] = "DejaVu Sans"
plt.rcParams["font.size"] = 10
plt.rcParams["axes.titlesize"] = 12
plt.rcParams["axes.labelsize"] = 11


def plot_class_distribution(df: pd.DataFrame, output_path: str) -> None:
    """Plot class distribution on linear and log scales to illustrate extreme imbalance."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    counts = df["Class"].value_counts()
    labels = ["Legitimate (0)", "Fraudulent (1)"]
    colors = ["#2b5c8f", "#d9534f"]

    # Linear count plot
    bars = axes[0].bar(labels, counts.values, color=colors, width=0.5, edgecolor="black", linewidth=1.2)
    axes[0].set_title("Class Frequency (Linear Scale)", fontweight="bold")
    axes[0].set_ylabel("Number of Transactions")
    for bar in bars:
        h = bar.get_height()
        axes[0].text(bar.get_x() + bar.get_width()/2., h + 2000, f"{h:,}\n({h/len(df)*100:.3f}%)",
                     ha="center", va="bottom", fontweight="bold")
    axes[0].set_ylim(0, max(counts.values) * 1.15)

    # Log scale plot to make minority class clearly visible
    bars_log = axes[1].bar(labels, counts.values, color=colors, width=0.5, edgecolor="black", linewidth=1.2)
    axes[1].set_yscale("log")
    axes[1].set_title("Class Frequency (Logarithmic Scale)", fontweight="bold")
    axes[1].set_ylabel("Number of Transactions (Log Scale)")
    for bar in bars_log:
        h = bar.get_height()
        axes[1].text(bar.get_x() + bar.get_width()/2., h * 1.3, f"{h:,}",
                     ha="center", va="bottom", fontweight="bold")

    plt.suptitle("Kaggle Credit Card Fraud Dataset - Extreme Class Imbalance (99.83% vs 0.17%)", fontsize=14, fontweight="bold")
    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=300)
    plt.close()


def plot_feature_correlations(df: pd.DataFrame, output_path: str, top_n: int = 10) -> None:
    """Plot top positive and negative feature correlations with Fraud Class."""
    corr = df.corr()["Class"].drop("Class").sort_values()
    top_neg = corr.head(top_n)
    top_pos = corr.tail(top_n)
    top_features = pd.concat([top_neg, top_pos])

    colors = ["#d9534f" if x < 0 else "#2b5c8f" for x in top_features.values]

    fig, ax = plt.subplots(figsize=(12, 7))
    bars = ax.barh(top_features.index, top_features.values, color=colors, edgecolor="black", linewidth=0.8)
    ax.axvline(0, color="black", linestyle="--", linewidth=1)
    ax.set_title(f"Top {top_n} Negative and Positive Feature Correlations with Fraud Class", fontweight="bold")
    ax.set_xlabel("Pearson Correlation Coefficient")
    ax.set_ylabel("Features")

    for bar in bars:
        w = bar.get_width()
        ha = "right" if w < 0 else "left"
        offset = -0.015 if w < 0 else 0.015
        ax.text(w + offset, bar.get_y() + bar.get_height()/2., f"{w:.3f}",
                va="center", ha=ha, fontsize=9, fontweight="bold")

    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=300)
    plt.close()


def plot_confusion_matrices(results: Dict[str, Dict[str, Any]], output_path: str) -> None:
    """Plot side-by-side annotated confusion matrices for all evaluated models."""
    n_models = len(results)
    cols = 2
    rows = (n_models + 1) // 2

    fig, axes = plt.subplots(rows, cols, figsize=(14, 5.5 * rows))
    axes = axes.flatten() if n_models > 1 else [axes]

    for idx, (name, res) in enumerate(results.items()):
        ax = axes[idx]
        cm = np.array(res["confusion_matrix"]["raw_array"])
        tn, fp, fn, tp = cm.ravel()
        total = cm.sum()

        labels = np.array([
            [f"TN\n{tn:,}\n({tn/total:.2%})", f"FP (Alarm)\n{fp:,}\n({fp/total:.2%})"],
            [f"FN (Miss)\n{fn:,}\n({fn/total:.2%})", f"TP (Fraud)\n{tp:,}\n({tp/total:.2%})"]
        ])

        sns.heatmap(cm, annot=labels, fmt="", cmap="Blues", cbar=False, ax=ax,
                    xticklabels=["Pred Legit", "Pred Fraud"],
                    yticklabels=["True Legit", "True Fraud"],
                    annot_kws={"size": 11, "weight": "bold"},
                    linewidths=1.5, linecolor="gray")

        fraud_r = res["class_1_fraud"]["recall_sensitivity"]
        fraud_p = res["class_1_fraud"]["precision"]
        fraud_f1 = res["class_1_fraud"]["f1_score"]
        ax.set_title(f"{name}\nFraud Recall: {fraud_r:.2%} | Precision: {fraud_p:.2%} | F1: {fraud_f1:.3f}",
                     fontweight="bold", fontsize=11)

    # Hide extra subplots if odd number
    for j in range(idx + 1, len(axes)):
        fig.delaxes(axes[j])

    plt.suptitle("Class-Wise Confusion Matrices Comparison on Test Set (56,962 Transactions)", fontsize=15, fontweight="bold")
    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=300)
    plt.close()


def plot_pr_and_roc_curves(
    y_true: np.ndarray,
    model_probs: Dict[str, np.ndarray],
    output_path: str,
) -> None:
    """Plot Precision-Recall Curves (PRC) and ROC Curves for all models."""
    fig, axes = plt.subplots(1, 2, figsize=(16, 6.5))

    # Baseline fraud prevalence
    prevalence = np.mean(y_true)

    colors = ["#2b5c8f", "#e67e22", "#27ae60", "#8e44ad"]

    # 1. Precision-Recall Curve (Paramount for Imbalanced Data)
    axes[0].axhline(y=prevalence, color="red", linestyle="--", linewidth=1.5,
                    label=f"No-Skill Baseline (Prevalence = {prevalence:.3%})")

    for (name, probs), color in zip(model_probs.items(), colors):
        precision, recall, _ = precision_recall_curve(y_true, probs)
        pr_auc = auc(recall, precision)
        axes[0].plot(recall, precision, label=f"{name} (PR-AUC = {pr_auc:.4f})",
                     linewidth=2.2, color=color)

    axes[0].set_title("Precision-Recall Curves (Key Metric for Imbalance)", fontweight="bold")
    axes[0].set_xlabel("Recall (Fraud Detection Rate)")
    axes[0].set_ylabel("Precision (Positive Predictive Value)")
    axes[0].set_ylim(-0.02, 1.05)
    axes[0].set_xlim(-0.02, 1.02)
    axes[0].legend(loc="lower left", frameon=True)
    axes[0].grid(True, linestyle="--", alpha=0.6)

    # 2. ROC Curve
    axes[1].plot([0, 1], [0, 1], color="navy", linestyle="--", linewidth=1.5, label="Random Guess (AUC = 0.5000)")

    for (name, probs), color in zip(model_probs.items(), colors):
        fpr, tpr, _ = roc_curve(y_true, probs)
        roc_auc = auc(fpr, tpr)
        axes[1].plot(fpr, tpr, label=f"{name} (ROC-AUC = {roc_auc:.4f})",
                     linewidth=2.2, color=color)

    axes[1].set_title("ROC Curves (False Positive Rate vs True Positive Rate)", fontweight="bold")
    axes[1].set_xlabel("False Positive Rate (1 - Specificity)")
    axes[1].set_ylabel("True Positive Rate (Sensitivity / Recall)")
    axes[1].set_ylim(-0.02, 1.05)
    axes[1].set_xlim(-0.02, 1.02)
    axes[1].legend(loc="lower right", frameon=True)
    axes[1].grid(True, linestyle="--", alpha=0.6)

    plt.suptitle("Model Evaluation Curves on Hold-out Test Set", fontsize=15, fontweight="bold")
    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=300)
    plt.close()


def plot_classwise_comparison(results: Dict[str, Dict[str, Any]], output_path: str) -> None:
    """Plot grouped bar charts of Class 0 vs Class 1 precision, recall, and F1 scores."""
    model_names = list(results.keys())

    # Extract metrics
    p0 = [results[m]["class_0_legitimate"]["precision"] for m in model_names]
    r0 = [results[m]["class_0_legitimate"]["recall_specificity"] for m in model_names]
    f0 = [results[m]["class_0_legitimate"]["f1_score"] for m in model_names]

    p1 = [results[m]["class_1_fraud"]["precision"] for m in model_names]
    r1 = [results[m]["class_1_fraud"]["recall_sensitivity"] for m in model_names]
    f1 = [results[m]["class_1_fraud"]["f1_score"] for m in model_names]

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Plot Class 1 (Fraud) - where the real challenge is!
    x = np.arange(len(model_names))
    width = 0.25

    axes[0].bar(x - width, p1, width, label="Precision (PPV)", color="#e74c3c", edgecolor="black")
    axes[0].bar(x, r1, width, label="Recall (Sensitivity)", color="#3498db", edgecolor="black")
    axes[0].bar(x + width, f1, width, label="F1-Score", color="#2ecc71", edgecolor="black")

    axes[0].set_title("Class 1 (Fraudulent) - Minority Class Metrics", fontweight="bold")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels([m.replace(" ", "\n") for m in model_names], fontsize=9)
    axes[0].set_ylabel("Score (0 to 1)")
    axes[0].set_ylim(0, 1.15)
    axes[0].legend(loc="upper left")
    axes[0].grid(axis="y", linestyle="--", alpha=0.7)

    # Add labels on F1 bars
    for i, val in enumerate(f1):
        axes[0].text(x[i] + width, val + 0.02, f"{val:.3f}", ha="center", fontsize=9, fontweight="bold")

    # Plot Missed Frauds (FN) vs False Alarms (FP)
    fns = [results[m]["confusion_matrix"]["fn"] for m in model_names]
    fps = [results[m]["confusion_matrix"]["fp"] for m in model_names]

    width2 = 0.35
    b1 = axes[1].bar(x - width2/2, fns, width2, label="Missed Frauds (FN - Direct Loss)", color="#c0392b", edgecolor="black")
    b2 = axes[1].bar(x + width2/2, fps, width2, label="False Alarms (FP - Customer Friction)", color="#f39c12", edgecolor="black")

    axes[1].set_title("Operational Trade-off: Missed Frauds vs False Alarms", fontweight="bold")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels([m.replace(" ", "\n") for m in model_names], fontsize=9)
    axes[1].set_ylabel("Transaction Count")
    axes[1].legend(loc="upper right")
    axes[1].set_yscale("log")
    axes[1].grid(axis="y", linestyle="--", alpha=0.7)

    for bar in b1:
        h = bar.get_height()
        axes[1].text(bar.get_x() + bar.get_width()/2., h * 1.15, f"{h}", ha="center", fontsize=8, fontweight="bold")
    for bar in b2:
        h = bar.get_height()
        axes[1].text(bar.get_x() + bar.get_width()/2., h * 1.15, f"{h}", ha="center", fontsize=8, fontweight="bold")

    plt.suptitle("Granular Class-Wise Performance & Operational Impact", fontsize=15, fontweight="bold")
    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=300)
    plt.close()


def plot_threshold_tuning(
    curve_data: Dict[str, Any],
    optimal_threshold: float,
    output_path: str,
    model_name: str = "XGBoost",
) -> None:
    """Plot Precision, Recall, and F1 curve across varying decision thresholds."""
    thresholds = curve_data["thresholds"]
    precisions = curve_data["precisions"]
    recalls = curve_data["recalls"]
    f1_scores = curve_data["f1_scores"]

    fig, ax = plt.subplots(figsize=(11, 6))
    ax.plot(thresholds, precisions, label="Precision (PPV)", color="#e74c3c", linewidth=2.2)
    ax.plot(thresholds, recalls, label="Recall (Sensitivity)", color="#3498db", linewidth=2.2)
    ax.plot(thresholds, f1_scores, label="F1-Score", color="#2ecc71", linewidth=2.5)

    # Highlight optimal threshold
    best_f1 = max(f1_scores)
    ax.axvline(optimal_threshold, color="black", linestyle="--", linewidth=1.5,
               label=f"Optimal F1 Threshold = {optimal_threshold:.2f} (F1 = {best_f1:.3f})")
    ax.scatter([optimal_threshold], [best_f1], color="black", s=70, zorder=5)

    # Highlight default 0.5 threshold
    idx_50 = int(np.argmin(np.abs(np.array(thresholds) - 0.5)))
    f1_50 = f1_scores[idx_50]
    ax.axvline(0.5, color="gray", linestyle=":", linewidth=1.5,
               label=f"Default Threshold = 0.50 (F1 = {f1_50:.3f})")

    ax.set_title(f"Decision Threshold Optimization for {model_name} (Fraud Class)", fontweight="bold")
    ax.set_xlabel("Classification Probability Threshold")
    ax.set_ylabel("Score (0 to 1)")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.05)
    ax.legend(loc="center right", frameon=True)
    ax.grid(True, linestyle="--", alpha=0.6)

    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=300)
    plt.close()


if __name__ == "__main__":
    print("Generating visualizations for Credit Card Fraud Detection...")
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_csv = os.path.join(base_dir, "data", "creditcard.csv")
    fig_dir = os.path.join(base_dir, "reports", "figures")
    os.makedirs(fig_dir, exist_ok=True)

    if os.path.exists(data_csv):
        print(f"Loading data from {data_csv}...")
        df_sample = pd.read_csv(data_csv)
        print("1. Plotting class distribution...")
        plot_class_distribution(df_sample, os.path.join(fig_dir, "class_distribution.png"))
        print("2. Plotting feature correlations...")
        plot_feature_correlations(df_sample, os.path.join(fig_dir, "feature_correlations.png"))
        print(f"Figures successfully generated in: {fig_dir}")
    else:
        print(f"Dataset not found at {data_csv}")
