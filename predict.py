"""Inference Module & CLI for Credit Card Fraud Detection.
Loads the trained model bundle and performs real-time scoring and classification.
"""

import os
import sys
import argparse
import joblib
import pandas as pd
import numpy as np

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(CURRENT_DIR)


def load_inference_pipeline(model_bundle_path: str = None):
    """Load the persisted model bundle containing model, preprocessor, and optimal threshold."""
    if model_bundle_path is None:
        model_bundle_path = os.path.join(CURRENT_DIR, "models", "best_fraud_model.joblib")

    if not os.path.exists(model_bundle_path):
        raise FileNotFoundError(f"Trained model bundle not found at: {model_bundle_path}. Run train_and_evaluate.py first.")

    bundle = joblib.load(model_bundle_path)
    return bundle


def predict_transaction(
    features_dict: dict,
    bundle: dict,
    custom_threshold: float = None,
) -> dict:
    """Score a single transaction dictionary.

    Args:
        features_dict: Dict with keys 'Time', 'V1' through 'V28', 'Amount'
        bundle: Loaded model bundle
        custom_threshold: Optional override for probability threshold

    Returns:
        Dict with fraud_probability, predicted_class, is_fraud, and alert_level.
    """
    model = bundle["model"]
    preprocessor = bundle["preprocessor"]
    threshold = custom_threshold if custom_threshold is not None else bundle["optimal_threshold"]

    df = pd.DataFrame([features_dict])
    df_scaled = preprocessor.transform(df)

    if hasattr(model, "predict_proba"):
        prob = float(model.predict_proba(df_scaled)[0, 1])
    else:
        dec = model.decision_function(df_scaled)[0]
        prob = 1.0 / (1.0 + np.exp(-dec))

    is_fraud = bool(prob >= threshold)

    if prob >= 0.8:
        risk_level = "CRITICAL RISK (Block Immediately)"
    elif prob >= threshold:
        risk_level = "HIGH RISK (Flag for 2FA Verification)"
    elif prob >= 0.2:
        risk_level = "MEDIUM RISK (Monitor)"
    else:
        risk_level = "LOW RISK (Approved)"

    return {
        "fraud_probability": round(prob, 4),
        "threshold_used": round(threshold, 4),
        "predicted_class": 1 if is_fraud else 0,
        "is_fraud": is_fraud,
        "risk_level": risk_level,
    }


def main():
    parser = argparse.ArgumentParser(description="Credit Card Fraud Inference Engine")
    parser.add_argument("--amount", type=float, default=149.99, help="Transaction amount in USD")
    parser.add_argument("--time", type=float, default=406.0, help="Elapsed seconds from first transaction")
    parser.add_argument("--threshold", type=float, default=None, help="Custom probability threshold")
    args = parser.parse_args()

    bundle = load_inference_pipeline()
    print("Loaded model:", bundle["model_name"])
    print(f"Optimal decision threshold: {bundle['optimal_threshold']:.3f}")

    # Create sample transaction with standard column ordering: Time, V1..V28, Amount
    sample_txn = {"Time": args.time}
    for i in range(1, 29):
        # Normal feature distribution
        sample_txn[f"V{i}"] = float(np.random.normal(0, 1))
    sample_txn["Amount"] = args.amount

    result = predict_transaction(sample_txn, bundle, custom_threshold=args.threshold)
    print("\n--- INFERENCE RESULT ---")
    print(f"Transaction Amount:  ${args.amount:.2f}")
    print(f"Fraud Probability:   {result['fraud_probability']:.2%}")
    print(f"Decision Threshold:  {result['threshold_used']:.2%}")
    print(f"Classification:      {'FRAUD DETECTED (Class 1)' if result['is_fraud'] else 'LEGITIMATE (Class 0)'}")
    print(f"Alert Status:        {result['risk_level']}")


if __name__ == "__main__":
    main()
