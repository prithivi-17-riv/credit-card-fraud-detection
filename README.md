# 💳 Credit Card Fraud Detection in Highly Imbalanced Data

An enterprise-grade Machine Learning system designed to identify fraudulent transactions under extreme class imbalance (~0.173% fraud rate) and evaluate performance using granular **class-wise metrics** and **cost-sensitive business thresholds**.

**Dataset**: [Kaggle – Credit Card Fraud Detection Dataset](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) (ULB Machine Learning Group)

---

## 📌 Executive Summary

Credit card fraud detection is characterized by **severe class imbalance**:
- **Total Transactions**: 284,807
- **Legitimate Transactions (Class 0)**: 284,315 (99.827%)
- **Fraudulent Transactions (Class 1)**: 492 (0.173%)
- **Imbalance Ratio**: **1 : 578**

### Why Overall Accuracy Fails
A naive "majority class" baseline that classifies **every transaction as legitimate** achieves an outstanding **99.827% accuracy**, yet catches **0% of fraudulent activity**. Therefore, overall accuracy is a completely deceptive metric. 

This project solves this by using **granular class-wise evaluation**:
- **Class 1 (Fraud)**: Precision (Positive Predictive Value), Recall (Sensitivity / Detection Rate), F1-Score, and False Negative Rate.
- **Class 0 (Legitimate)**: Precision (NPV), Recall (Specificity), and False Positive Rate.
- **Imbalanced Benchmarks**: Area Under the Precision-Recall Curve (**PR-AUC** / Average Precision) and Matthews Correlation Coefficient (**MCC**).

---

## 📊 Class-Wise Performance Results (Test Set: 56,962 Transactions)

| Model Architecture | Legit Prec (C0) | Legit Recall (C0) | Fraud Prec (C1) | Fraud Recall (C1) | Fraud F1 | PR-AUC | ROC-AUC | Balanced Acc | TP (Caught) | FN (Missed) | FP (False Alarms) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Cost-Sensitive Logistic Reg.** | 0.9999 | 0.9757 | 6.11% | **91.84%** | 0.1146 | 0.7111 | 0.9720 | 94.70% | **90** | 8 | 1,382 |
| **SMOTE + Logistic Regression** | 0.9998 | 0.9972 | 35.37% | 88.78% | 0.5058 | 0.7513 | 0.9675 | 94.25% | 87 | 11 | 159 |
| **Balanced Random Forest** | 0.9997 | **0.9997** | 83.16% | 80.61% | 0.8187 | 0.8316 | 0.9764 | 90.29% | 79 | 19 | **16** |
| **XGBoost (Cost-Weighted)** *(Default)* | 0.9998 | 0.9993 | 66.67% | 85.71% | 0.7500 | **0.8525** | **0.9819** | 92.82% | 84 | 14 | 42 |
| **XGBoost (Optimal Threshold = 0.98)** | **0.9996** | **0.9999** | **92.77%** | 78.57% | **0.8508** | **0.8525** | **0.9819** | 89.28% | 77 | 21 | **6** |

---

## 🎯 Key Engineering Highlights

### 1. Strict Featurization Ordering
Adheres to the gold standard ML pipeline principle:
- **Never fit scalers or resamplers on the full dataset.**
- Stratified 80/20 train/test split is executed first.
- `RobustScaler` (median & interquartile range) is fitted **strictly on training data** to handle heavy transaction amount outliers (up to $25k) without distorting test evaluation.
- `V1` to `V28` (PCA components) retain their mathematical zero-mean orthogonal properties.

### 2. Multi-Paradigm Imbalance Handling
- **Cost-Sensitive Learning**: Penalizes misclassifying the minority class inversely to class frequency ($w_1 = 578 \times w_0$).
- **SMOTE Pipeline**: Synthetic Minority Over-sampling applied exclusively inside `imblearn.pipeline.Pipeline` on train partitions, preventing information leakage.
- **Ensemble Bagging**: `BalancedRandomForestClassifier` with balanced subsampling.
- **Gradient Boosted Trees**: `XGBClassifier` with `scale_pos_weight=577.9` optimizing the Precision-Recall objective.

### 3. Decision Threshold Optimization
Because positive class weights shift probability estimates, the default 0.5 threshold is suboptimal. 
- Threshold optimization reveals the optimal operating point at **$T = 0.98$**, boosting Fraud Precision from **66.67% to 92.77%** and reducing False Alarms from **42 down to just 6**, achieving a peak Fraud F1 of **0.851**.

---

## 🚀 Quick Start Guide

### 1. Installation
Activate your Python environment and install the required dependencies:
```bash
pip install -r requirements.txt
```

### 2. Train and Evaluate the Full Pipeline
Executes the end-to-end pipeline, trains all 4 models, computes class-wise metrics, exports 6 high-resolution charts, and persists model artifacts:
```bash
python train_and_evaluate.py
```

### 3. Launch the Interactive Dashboard
Launch the Streamlit web dashboard to interact with the threshold simulator, inspect class-wise trade-offs, and test real-time fraud scoring:
```bash
streamlit run fraud_dashboard.py
```

### 4. Run CLI Real-Time Inference
Score individual transactions and receive fraud probabilities, decision flags, and alert levels:
```bash
python predict.py --amount 250.00 --time 45000.0
```

---

## 📁 Repository Structure

```
credit_card_fraud_detection/
├── data/
│   └── creditcard.csv                 # Kaggle dataset (284,807 transactions)
├── models/
│   ├── best_fraud_model.joblib        # Persisted best model bundle (XGBoost + threshold + scaler)
│   └── preprocessor.joblib            # Fitted RobustScaler preprocessor
├── reports/
│   ├── figures/
│   │   ├── class_distribution.png     # Linear & logarithmic class imbalance charts
│   │   ├── feature_correlations.png   # Top positive & negative feature correlations
│   │   ├── confusion_matrices.png     # Side-by-side annotated confusion matrices
│   │   ├── pr_and_roc_curves.png      # Precision-Recall & ROC curves with baselines
│   │   ├── classwise_metrics_comparison.png # Grouped bar chart comparing C0 and C1
│   │   └── threshold_tuning.png       # Decision threshold vs Precision/Recall/F1
│   └── evaluation_summary.json        # Machine-readable evaluation report
├── src/
│   ├── __init__.py
│   ├── data_loader.py                 # Data loading, validation, and stratified splitting
│   ├── preprocessing.py               # RobustScaler with strict featurization ordering
│   ├── models.py                      # Model definitions (Cost-sensitive, SMOTE, Ensemble, XGBoost)
│   ├── evaluation.py                  # Class-wise metrics, confusion matrix, threshold tuning
│   └── visualization.py               # Matplotlib & Seaborn high-res figure generation
├── Credit_Card_Fraud_Detection_Notebook.ipynb # Storytelling Jupyter Notebook
├── train_and_evaluate.py              # Main training and evaluation CLI runner
├── predict.py                         # Single transaction scoring CLI
├── fraud_dashboard.py                 # Interactive Streamlit web application
├── requirements.txt                   # Python dependencies
└── README.md                          # Comprehensive project documentation
```

---

## 📈 Visual Reports Generated

All charts are saved at 300 DPI in `reports/figures/`:
1. `class_distribution.png`: Demonstrates the 99.83% vs 0.17% distribution across linear and logarithmic scales.
2. `feature_correlations.png`: Highlights `V17`, `V14`, `V12`, `V10` as strong negative indicators, and `V4`, `V11` as positive indicators.
3. `confusion_matrices.png`: Full $2 \times 2$ matrices with raw counts and normalized percentage breakdowns.
4. `pr_and_roc_curves.png`: Compares PR curves against the 0.17% no-skill line, proving XGBoost achieves 0.8525 PR-AUC.
5. `classwise_metrics_comparison.png`: Visualizes the financial trade-off between missed fraud (loss) and false alarms (friction).
6. `threshold_tuning.png`: Precision, Recall, and F1 trajectories across the complete threshold spectrum ($0.01 \le T \le 0.99$).
