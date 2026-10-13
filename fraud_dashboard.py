"""Interactive Streamlit Dashboard for Credit Card Fraud Detection.
Provides exploratory analytics, model comparison, class-wise evaluation metrics,
an interactive decision threshold simulator, and real-time transaction scoring.
"""

import os
import sys
import json
import joblib
import pandas as pd
import numpy as np
import streamlit as st

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(CURRENT_DIR)

st.set_page_config(
    page_title="Credit Card Fraud Detection System",
    page_icon="💳",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# Custom CSS for modern styling + mobile responsiveness
st.markdown("""
<style>
    /* ── Base Styles ── */
    .main-header {
        font-size: 2.2rem;
        font-weight: 800;
        color: #1E293B;
        margin-bottom: 0.2rem;
        line-height: 1.2;
    }
    .sub-header {
        font-size: 1.05rem;
        color: #64748B;
        margin-bottom: 1.5rem;
        line-height: 1.5;
    }
    .metric-card {
        background-color: #F8FAFC;
        border: 1px solid #E2E8F0;
        border-radius: 10px;
        padding: 1.2rem;
        text-align: center;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
        margin-bottom: 0.5rem;
    }
    .metric-value {
        font-size: 1.8rem;
        font-weight: 700;
        color: #0F172A;
    }
    .metric-label {
        font-size: 0.85rem;
        font-weight: 600;
        color: #64748B;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    .fraud-badge {
        background-color: #FEE2E2;
        color: #991B1B;
        padding: 4px 10px;
        border-radius: 6px;
        font-weight: 700;
        font-size: 0.9rem;
        display: inline-block;
    }
    .legit-badge {
        background-color: #DCFCE7;
        color: #166534;
        padding: 4px 10px;
        border-radius: 6px;
        font-weight: 700;
        font-size: 0.9rem;
        display: inline-block;
    }

    /* ── Tablet: ≤ 768px ── */
    @media (max-width: 768px) {
        .main-header {
            font-size: 1.5rem;
        }
        .sub-header {
            font-size: 0.9rem;
            margin-bottom: 1rem;
        }
        .metric-value {
            font-size: 1.4rem;
        }
        /* Make Streamlit columns wrap into 2-column grid */
        [data-testid="stHorizontalBlock"] {
            flex-wrap: wrap !important;
        }
        [data-testid="stHorizontalBlock"] > [data-testid="stColumn"] {
            min-width: calc(50% - 0.5rem) !important;
            flex: 1 1 calc(50% - 0.5rem) !important;
        }
    }

    /* ── Mobile: ≤ 480px ── */
    @media (max-width: 480px) {
        .main-header {
            font-size: 1.25rem;
        }
        .sub-header {
            font-size: 0.82rem;
        }
        .metric-value {
            font-size: 1.2rem;
        }
        .metric-card {
            padding: 0.85rem;
        }
        /* Stack all columns to full width on phones */
        [data-testid="stHorizontalBlock"] > [data-testid="stColumn"] {
            min-width: 100% !important;
            flex: 1 1 100% !important;
        }
        /* Tab labels: allow wrapping */
        [data-testid="stTabs"] [role="tab"] {
            font-size: 0.75rem;
            padding: 0.4rem 0.5rem;
        }
        /* Compact number inputs and sliders */
        [data-testid="stSlider"],
        [data-testid="stNumberInput"] {
            font-size: 0.85rem;
        }
        /* Ensure images don't overflow */
        img {
            max-width: 100%;
            height: auto;
        }
        /* Button full-width on mobile */
        [data-testid="stButton"] button {
            width: 100%;
        }
    }

    /* ── General: prevent horizontal scroll ── */
    .block-container {
        padding-left: 1rem !important;
        padding-right: 1rem !important;
        max-width: 100% !important;
        overflow-x: hidden;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_bundle():
    bundle_path = os.path.join(CURRENT_DIR, "models", "best_fraud_model.joblib")
    if os.path.exists(bundle_path):
        return joblib.load(bundle_path)
    return None


@st.cache_data
def load_summary():
    summary_path = os.path.join(CURRENT_DIR, "reports", "evaluation_summary.json")
    if os.path.exists(summary_path):
        with open(summary_path, "r") as f:
            return json.load(f)
    return None


bundle = load_bundle()
summary = load_summary()

# Top Header
st.markdown('<div class="main-header">💳 Credit Card Fraud Detection System</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Machine Learning System for Highly Imbalanced Data with Class-Wise Performance Metrics</div>', unsafe_allow_html=True)

# Key Dataset KPIs
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.markdown("""
    <div class="metric-card">
        <div class="metric-label">Total Transactions</div>
        <div class="metric-value">284,807</div>
    </div>
    """, unsafe_allow_html=True)
with col2:
    st.markdown("""
    <div class="metric-card">
        <div class="metric-label">Legitimate (Class 0)</div>
        <div class="metric-value" style="color: #166534;">99.827%</div>
    </div>
    """, unsafe_allow_html=True)
with col3:
    st.markdown("""
    <div class="metric-card">
        <div class="metric-label">Fraudulent (Class 1)</div>
        <div class="metric-value" style="color: #991B1B;">0.173% (492)</div>
    </div>
    """, unsafe_allow_html=True)
with col4:
    st.markdown("""
    <div class="metric-card">
        <div class="metric-label">Imbalance Ratio</div>
        <div class="metric-value" style="color: #4338CA;">1 : 578</div>
    </div>
    """, unsafe_allow_html=True)

st.write("")

# Tabs
tab1, tab2, tab3, tab4 = st.tabs([
    "📊 Class-Wise Model Evaluation",
    "📈 Imbalance & EDA Insights",
    "🎛️ Decision Threshold Simulator",
    "🔍 Real-Time Transaction Scoring",
])

# ----------------- TAB 1: MODEL EVALUATION -----------------
with tab1:
    st.subheader("Model Comparison on Hold-out Test Set (56,962 Transactions)")
    st.markdown("""
    Under extreme class imbalance, standard accuracy is misleading (a naive classifier predicting 100% legitimate achieves 99.83% accuracy while catching zero fraud).
    Evaluation must focus on **granular class-wise metrics**, especially **Fraud Precision**, **Fraud Recall (Sensitivity)**, and **PR-AUC**.
    """)

    if summary and "models_evaluation" in summary:
        models_data = summary["models_evaluation"]
        rows = []
        for name, data in models_data.items():
            c0 = data["class_0_legitimate"]
            c1 = data["class_1_fraud"]
            ov = data["overall_summary"]
            cm = data["confusion_matrix"]
            rows.append({
                "Model Architecture": name,
                "Fraud Recall (Caught)": f"{c1['recall_sensitivity']:.2%}",
                "Fraud Precision (Accuracy)": f"{c1['precision']:.2%}",
                "Fraud F1-Score": f"{c1['f1_score']:.4f}",
                "PR-AUC (Avg Precision)": f"{ov['pr_auc']:.4f}",
                "ROC-AUC": f"{ov['roc_auc']:.4f}",
                "Balanced Acc": f"{ov['balanced_accuracy']:.2%}",
                "True Positives (TP)": cm["tp"],
                "Missed Frauds (FN)": cm["fn"],
                "False Alarms (FP)": cm["fp"],
                "Legit Recall (C0)": f"{c0['recall_specificity']:.2%}",
            })
        st.dataframe(pd.DataFrame(rows), use_container_width=True)

    st.write("---")
    st.subheader("Evaluation Visualizations")

    fig_cols = st.columns(2)
    with fig_cols[0]:
        cm_path = os.path.join(CURRENT_DIR, "reports", "figures", "confusion_matrices.png")
        if os.path.exists(cm_path):
            st.image(cm_path, caption="Confusion Matrices Comparison (Hold-Out Test Set)")
    with fig_cols[1]:
        pr_path = os.path.join(CURRENT_DIR, "reports", "figures", "pr_and_roc_curves.png")
        if os.path.exists(pr_path):
            st.image(pr_path, caption="Precision-Recall & ROC Curves")

    bar_path = os.path.join(CURRENT_DIR, "reports", "figures", "classwise_metrics_comparison.png")
    if os.path.exists(bar_path):
        st.image(bar_path, caption="Class-Wise Metric Breakdown & Operational Trade-Offs (Missed Frauds vs False Alarms)")

# ----------------- TAB 2: IMBALANCE & EDA -----------------
with tab2:
    st.subheader("Exploratory Data Analysis & Imbalance Characterization")
    st.markdown("""
    The dataset consists of **284,807 transactions** over 2 days in September 2013 by European cardholders.
    All features **V1 to V28** are principal components obtained via PCA due to confidentiality constraints.
    **Time** (seconds elapsed) and **Amount** (transaction value) are the only raw continuous features.
    """)

    eda_cols = st.columns(2)
    with eda_cols[0]:
        dist_path = os.path.join(CURRENT_DIR, "reports", "figures", "class_distribution.png")
        if os.path.exists(dist_path):
            st.image(dist_path, caption="Severe Class Imbalance: Linear vs Logarithmic Scale")
    with eda_cols[1]:
        corr_path = os.path.join(CURRENT_DIR, "reports", "figures", "feature_correlations.png")
        if os.path.exists(corr_path):
            st.image(corr_path, caption="Top Correlated PCA Features with Fraudulent Transactions")

    st.info("""
    **Key Insights from EDA:**
    1. **Extreme Imbalance**: Only 492 frauds out of 284,807 transactions (0.173%).
    2. **Strongest Fraud Predictors**: Features **V17, V14, V12, and V10** show strong negative correlations with fraud (fraudulent transactions exhibit significantly lower values). Conversely, **V4 and V11** exhibit positive correlations.
    3. **Transaction Amount Skew**: Legitimate transactions average $88.29 with a maximum of $25,691. Fraudulent transactions average $122.21 with a maximum of $2,125.87. Robust scaling prevents outlier distortion.
    """)

# ----------------- TAB 3: THRESHOLD SIMULATOR -----------------
with tab3:
    st.subheader("🎛️ Interactive Decision Threshold Simulator")
    st.markdown("""
    By default, binary classifiers predict fraud if $P(\\text{Fraud}) \\ge 0.50$.
    However, under severe imbalance, adjusting the decision threshold allows banks to balance **fraud detection coverage** against **customer friction from false alarms**.
    """)

    opt_thresh = bundle["optimal_threshold"] if bundle else 0.98

    sim_col1, sim_col2 = st.columns([1, 2])
    with sim_col1:
        threshold = st.slider(
            "Classification Probability Threshold",
            min_value=0.01,
            max_value=0.99,
            value=float(opt_thresh),
            step=0.01,
            help="Transactions with predicted probability >= threshold will be flagged as fraud.",
        )
        st.write(f"**Selected Threshold:** `{threshold:.2f}` (Optimal F1: `{opt_thresh:.2f}`)")

        cost_missed_fraud = st.number_input("Average Cost of Missed Fraud ($)", value=250.0, step=25.0)
        cost_false_alarm = st.number_input("Cost of False Alarm / Investigation ($)", value=10.0, step=2.0)

    with sim_col2:
        thresh_img = os.path.join(CURRENT_DIR, "reports", "figures", "threshold_tuning.png")
        if os.path.exists(thresh_img):
            st.image(thresh_img, caption="Precision, Recall, and F1-Score Trade-off Curve")

    st.markdown("### Business Strategy Recommendation:")
    st.markdown(f"""
    - **Conservative Risk Strategy (Low Threshold ~0.20 - 0.50)**: Prioritizes catching nearly every fraud (Recall > 90%), but generates more false positives requiring customer confirmation SMS/2FA.
    - **Optimized Balanced Strategy (Threshold ~{opt_thresh:.2f})**: Maximizes the F1-Score, yielding **92.8% Precision** and **78.6% Recall**, drastically cutting false alarms.
    - **High-Friction Sensitivity Strategy (High Threshold > 0.98)**: Only blocks transactions with overwhelming confidence to minimize customer checkout abandonment.
    """)

# ----------------- TAB 4: REAL-TIME INFERENCE -----------------
with tab4:
    st.subheader("🔍 Real-Time Transaction Scoring Playground")
    st.markdown("Test the production fraud scoring pipeline on synthetic or real transaction vectors.")

    preset = st.selectbox(
        "Select Transaction Profile Preset",
        ["Custom Input", "Typical Legitimate Coffee Purchase", "Suspicious Large Electronic Transfer", "Card-Testing Anomaly"]
    )

    col_a, col_b = st.columns(2)
    with col_a:
        if preset == "Typical Legitimate Coffee Purchase":
            t_amount = 14.50
            t_time = 3600.0
            v14_val = 0.20
            v17_val = -0.10
            v12_val = 0.05
            v4_val = -0.15
        elif preset == "Suspicious Large Electronic Transfer":
            t_amount = 1850.00
            t_time = 72000.0
            v14_val = -7.50
            v17_val = -6.80
            v12_val = -5.40
            v4_val = 4.20
        elif preset == "Card-Testing Anomaly":
            t_amount = 1.00
            t_time = 15000.0
            v14_val = -4.20
            v17_val = -3.90
            v12_val = -3.10
            v4_val = 3.10
        else:
            t_amount = 120.00
            t_time = 50000.0
            v14_val = 0.0
            v17_val = 0.0
            v12_val = 0.0
            v4_val = 0.0

        amount = st.number_input("Transaction Amount ($)", value=float(t_amount), min_value=0.0)
        txn_time = st.number_input("Elapsed Seconds (Time)", value=float(t_time), min_value=0.0)

    with col_b:
        v14 = st.slider("PCA Feature V14 (Strong Negative Correlator)", -15.0, 5.0, float(v14_val), 0.1)
        v17 = st.slider("PCA Feature V17 (Strong Negative Correlator)", -15.0, 5.0, float(v17_val), 0.1)
        v12 = st.slider("PCA Feature V12 (Strong Negative Correlator)", -15.0, 5.0, float(v12_val), 0.1)
        v4 = st.slider("PCA Feature V4 (Strong Positive Correlator)", -5.0, 15.0, float(v4_val), 0.1)

    if st.button("🚀 Score Transaction", type="primary"):
        if bundle:
            # Construct feature dict
            txn_dict = {"Time": txn_time}
            for i in range(1, 29):
                txn_dict[f"V{i}"] = 0.0
            txn_dict["V14"] = v14
            txn_dict["V17"] = v17
            txn_dict["V12"] = v12
            txn_dict["V4"] = v4
            txn_dict["Amount"] = amount

            from predict import predict_transaction
            res = predict_transaction(txn_dict, bundle)

            st.write("---")
            res_col1, res_col2 = st.columns(2)
            with res_col1:
                st.metric("Fraud Probability Score", f"{res['fraud_probability']:.2%}")
                st.metric("Decision Threshold Used", f"{res['threshold_used']:.2%}")
            with res_col2:
                if res["is_fraud"]:
                    st.markdown('<div class="fraud-badge" style="font-size: 1.2rem; display: inline-block;">⚠️ FRAUD SUSPECTED</div>', unsafe_allow_html=True)
                else:
                    st.markdown('<div class="legit-badge" style="font-size: 1.2rem; display: inline-block;">✅ TRANSACTION APPROVED</div>', unsafe_allow_html=True)
                st.write(f"**System Action:** {res['risk_level']}")
        else:
            st.error("Model bundle not found. Please run train_and_evaluate.py first.")
