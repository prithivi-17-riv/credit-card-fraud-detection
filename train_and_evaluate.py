"""Credit Card Fraud Detection - Main Training & Evaluation Pipeline.
Implements complete end-to-end ML workflow for highly imbalanced data:
1. Data loading and validation
2. Exploratory Data Analysis & visual generation
3. Strict featurization ordering (stratified train/test split, then RobustScaler)
4. Model training across 4 paradigms (Cost-sensitive, SMOTE, Ensemble, Boosted trees)
5. Comprehensive class-wise metrics evaluation
6. Decision threshold optimization & business trade-off analysis
7. Publication-ready figure export and artifact persistence
"""

import os
import sys
import json
import time
import pandas as pd
import numpy as np
import joblib

# Ensure src package is importable
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(CURRENT_DIR)

from src.data_loader import load_dataset, validate_dataset, split_data
from src.preprocessing import preprocess_features, FraudDataPreprocessor
from src.models import get_models
from src.evaluation import (
    compute_classwise_metrics,
    optimize_threshold,
    format_metrics_table,
)
from src.visualization import (
    plot_class_distribution,
    plot_feature_correlations,
    plot_confusion_matrices,
    plot_pr_and_roc_curves,
    plot_classwise_comparison,
    plot_threshold_tuning,
)


def main():
    print("=" * 80)
    print("CREDIT CARD FRAUD DETECTION IN HIGHLY IMBALANCED DATA")
    print("Machine Learning System & Class-Wise Performance Evaluation")
    print("=" * 80)

    data_path = os.path.join(CURRENT_DIR, "data", "creditcard.csv")
    models_dir = os.path.join(CURRENT_DIR, "models")
    reports_dir = os.path.join(CURRENT_DIR, "reports")
    figures_dir = os.path.join(reports_dir, "figures")

    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(figures_dir, exist_ok=True)

    # 1. Load and Validate Dataset
    print(f"\n[1/6] Loading dataset from: {data_path} ...")
    start_time = time.time()
    df = load_dataset(data_path)
    stats = validate_dataset(df)
    print(f"      Loaded {stats['total_records']:,} transactions with {stats['total_features']} features.")
    print(f"      Class 0 (Legitimate): {stats['class_0_legit']:,} ({100 - stats['fraud_percentage']:.3f}%)")
    print(f"      Class 1 (Fraudulent): {stats['class_1_fraud']:,} ({stats['fraud_percentage']:.3f}%)")
    print(f"      Imbalance Ratio:      {stats['imbalance_ratio']}")
    print(f"      Total Null Values:    {stats['total_nulls']}")

    # 2. Generate EDA Visualizations
    print(f"\n[2/6] Generating EDA visualizations...")
    dist_fig = os.path.join(figures_dir, "class_distribution.png")
    corr_fig = os.path.join(figures_dir, "feature_correlations.png")
    plot_class_distribution(df, dist_fig)
    plot_feature_correlations(df, corr_fig)
    print(f"      Saved: {dist_fig}")
    print(f"      Saved: {corr_fig}")

    # 3. Stratified Train-Test Split & Strict Preprocessing Ordering
    print(f"\n[3/6] Performing Stratified Train-Test Split (80% Train, 20% Test)...")
    X_train, X_test, y_train, y_test = split_data(df, test_size=0.2, random_state=42)
    print(f"      Train Set: {len(X_train):,} transactions (Fraud: {int(y_train.sum())})")
    print(f"      Test Set:  {len(X_test):,} transactions (Fraud: {int(y_test.sum())})")

    print(f"      Fitting RobustScaler on Train Set (strict featurization ordering)...")
    X_train_scaled, X_test_scaled, preprocessor = preprocess_features(X_train, X_test)
    preprocessor_path = os.path.join(models_dir, "preprocessor.joblib")
    preprocessor.save(preprocessor_path)
    print(f"      Saved fitted preprocessor to: {preprocessor_path}")

    # 4. Train Models
    imbalance_ratio = float(stats["class_0_legit"] / stats["class_1_fraud"])
    models = get_models(imbalance_ratio=imbalance_ratio, random_state=42)

    results = {}
    model_probs = {}
    fitted_models = {}

    print(f"\n[4/6] Training {len(models)} Imbalanced Learning Models...")
    for name, model in models.items():
        t0 = time.time()
        print(f"      Training: '{name}' ...", end="", flush=True)
        model.fit(X_train_scaled, y_train)
        train_time = time.time() - t0

        # Predictions
        y_pred = model.predict(X_test_scaled)
        if hasattr(model, "predict_proba"):
            y_prob = model.predict_proba(X_test_scaled)[:, 1]
        elif hasattr(model, "decision_function"):
            decision = model.decision_function(X_test_scaled)
            y_prob = (decision - decision.min()) / (decision.max() - decision.min())
        else:
            y_prob = None

        if y_prob is not None:
            model_probs[name] = y_prob

        fitted_models[name] = model

        # Metrics
        res = compute_classwise_metrics(y_test.values, y_pred, y_prob)
        res["training_time_seconds"] = round(train_time, 2)
        results[name] = res

        fraud_r = res["class_1_fraud"]["recall_sensitivity"]
        fraud_p = res["class_1_fraud"]["precision"]
        fraud_f1 = res["class_1_fraud"]["f1_score"]
        pr_auc = res["overall_summary"]["pr_auc"]
        print(f" Done ({train_time:.1f}s) | Fraud Recall: {fraud_r:.2%} | Precision: {fraud_p:.2%} | F1: {fraud_f1:.3f} | PR-AUC: {pr_auc:.4f}")

    # 5. Format & Present Class-Wise Evaluation Metrics Table
    print(f"\n[5/6] Comprehensive Class-Wise Performance Evaluation Table:")
    table_df = format_metrics_table(results)
    print("-" * 120)
    print(table_df.to_string(index=False))
    print("-" * 120)

    # Export figures
    print(f"\n      Exporting evaluation charts...")
    cm_fig = os.path.join(figures_dir, "confusion_matrices.png")
    curves_fig = os.path.join(figures_dir, "pr_and_roc_curves.png")
    bar_fig = os.path.join(figures_dir, "classwise_metrics_comparison.png")

    plot_confusion_matrices(results, cm_fig)
    plot_pr_and_roc_curves(y_test.values, model_probs, curves_fig)
    plot_classwise_comparison(results, bar_fig)

    print(f"      Saved: {cm_fig}")
    print(f"      Saved: {curves_fig}")
    print(f"      Saved: {bar_fig}")

    # 6. Select Best Model & Perform Threshold Optimization
    # Rank by Fraud F1-Score and PR-AUC
    best_model_name = max(results.keys(), key=lambda m: (results[m]["overall_summary"]["pr_auc"] or 0, results[m]["class_1_fraud"]["f1_score"]))
    best_model = fitted_models[best_model_name]
    best_probs = model_probs[best_model_name]

    print(f"\n[6/6] Threshold Tuning & Business Impact Analysis for Top Model: '{best_model_name}'...")
    threshold_res = optimize_threshold(y_test.values, best_probs)
    opt_t = threshold_res["optimal_threshold"]
    best_fraud_f1 = threshold_res["best_fraud_f1"]
    opt_p = threshold_res["best_metrics"]["precision"]
    opt_r = threshold_res["best_metrics"]["recall"]

    thresh_fig = os.path.join(figures_dir, "threshold_tuning.png")
    plot_threshold_tuning(threshold_res["curve_data"], opt_t, thresh_fig, model_name=best_model_name)
    print(f"      Optimal Decision Threshold: {opt_t:.3f}")
    print(f"      At Optimal Threshold -> Precision: {opt_p:.2%} | Recall: {opt_r:.2%} | F1: {best_fraud_f1:.3f}")
    print(f"      Saved: {thresh_fig}")

    # Save Best Model Artifacts
    best_model_path = os.path.join(models_dir, "best_fraud_model.joblib")
    joblib.dump({
        "model_name": best_model_name,
        "model": best_model,
        "optimal_threshold": opt_t,
        "preprocessor": preprocessor,
    }, best_model_path)
    print(f"      Persisted best model bundle to: {best_model_path}")

    # Save JSON Summary
    summary_path = os.path.join(reports_dir, "evaluation_summary.json")
    with open(summary_path, "w") as f:
        json.dump({
            "dataset_stats": stats,
            "models_evaluation": results,
            "best_model": {
                "name": best_model_name,
                "optimal_threshold": opt_t,
                "best_metrics": threshold_res["best_metrics"],
            }
        }, f, indent=2)
    print(f"      Persisted JSON evaluation summary to: {summary_path}")

    total_time = time.time() - start_time
    print(f"\nPipeline successfully completed in {total_time:.1f} seconds!")
    print("=" * 80)


if __name__ == "__main__":
    main()
