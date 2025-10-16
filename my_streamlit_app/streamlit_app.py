import streamlit as st
import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_curve, precision_recall_curve, roc_auc_score
)
import matplotlib.pyplot as plt
from PIL import Image

# Page config
st.set_page_config(page_title="Fraud Model Explainer", layout="wide")

from PIL import Image

# Sidebar — logo in top-left corner
with st.sidebar:
    logo = Image.open("assets/banner.png")
    st.image(logo, width=250)  # adjust width as needed


# Load models and feature names
@st.cache_resource
def load_models_and_features(models_dir="models"):
    models = {}
    for p in Path(models_dir).glob("*.pkl"):
        name = p.stem
        if name == "feature_names":
            continue
        models[name] = joblib.load(p)
    feature_names = joblib.load(Path(models_dir) / "feature_names.pkl")
    return models, feature_names

models_dict, feature_names = load_models_and_features()

# Sidebar — model selector
st.sidebar.header("Model Selection")
model_name = st.sidebar.selectbox("Choose a model", list(models_dict.keys()))
model = models_dict[model_name]

# Sidebar — input features
st.sidebar.header("Input Features")
selected_features = {
    "Transaction_Amount": st.sidebar.number_input("Transaction Amount", value=0.0),
    "Account_Balance": st.sidebar.number_input("Account Balance", value=0.0),
    "Previous_Fraudulent_Activity": int(st.sidebar.checkbox("Previous Fraudulent Activity")),
    "Daily_Transaction_Count": st.sidebar.number_input("Daily Transaction Count", value=0),
    "Risk_Score": st.sidebar.slider("Risk Score", 0.0, 1.0, 0.5),
    "Is_Weekend": int(st.sidebar.checkbox("Is Weekend"))
}

# Sidebar — decision threshold
threshold = st.sidebar.slider("Decision threshold", 0.0, 1.0, 0.5)

# Build input row
X_user_full = pd.DataFrame(columns=feature_names)
X_user_full.loc[0] = 0.0
for feat, val in selected_features.items():
    X_user_full.at[0, feat] = float(val)
X_user = X_user_full.copy()

# Display user inputs
st.subheader("User Inputs")
st.dataframe(X_user[selected_features.keys()])

# Explain Risk_Score
with st.expander("What is Risk_Score?"):
    st.markdown("""
    **Risk_Score** is a precomputed feature from upstream systems. It reflects how suspicious a transaction looks based on:
    - Velocity checks (too many transactions)
    - Device/IP anomalies
    - Rule-based flags
    - Historical behavior

    Values range from `0.0` (low risk) to `1.0` (high risk). Your model uses this to boost fraud sensitivity.
    """)

# Run prediction
proba = model.predict_proba(X_user)[0, 1]
label = "Fraud" if proba >= threshold else "Legit"

st.subheader("Prediction")
st.metric("Fraud Probability", f"{proba:.2%}")
st.metric("Predicted Class", label)

# Load test set
@st.cache_data
def load_test_set(path="models/test_set.csv"):
    df = pd.read_csv(path)
    X_test = df[feature_names]
    y_test = df["Fraud_Label"]
    return df, X_test, y_test

try:
    df_test, X_test, y_test = load_test_set()
    probs = model.predict_proba(X_test)[:, 1]
    y_pred = (probs >= threshold).astype(int)

    # Metrics
    st.subheader("Model Performance on Test Set")
    metrics = {
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred),
        "Recall": recall_score(y_test, y_pred),
        "F1 Score": f1_score(y_test, y_pred)
    }
    st.table(pd.DataFrame(metrics, index=["Score"]).T)

    # Compact confusion matrix plot
    st.subheader("Confusion Matrix")
    cm = confusion_matrix(y_test, y_pred)
    fig_cm, ax_cm = plt.subplots(figsize=(2, 2))
    ax_cm.imshow(cm, cmap="Blues")
    ax_cm.set_xticks([0, 1])
    ax_cm.set_yticks([0, 1])
    ax_cm.set_xticklabels(["Legit", "Fraud"])
    ax_cm.set_yticklabels(["Legit", "Fraud"])
    ax_cm.set_xlabel("Predicted")
    ax_cm.set_ylabel("Actual")
    for (i, j), val in np.ndenumerate(cm):
        ax_cm.text(j, i, str(val), ha="center", va="center", fontsize=6)
    st.pyplot(fig_cm)

    # ROC and PR curves
    fpr, tpr, _ = roc_curve(y_test, probs)
    precision, recall, _ = precision_recall_curve(y_test, probs)
    auc_score = roc_auc_score(y_test, probs)

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("ROC Curve")
        fig_roc, ax_roc = plt.subplots()
        ax_roc.plot(fpr, tpr, label=f"AUC = {auc_score:.3f}")
        ax_roc.plot([0, 1], [0, 1], "--", color="gray")
        ax_roc.set_xlabel("False Positive Rate")
        ax_roc.set_ylabel("True Positive Rate")
        ax_roc.legend()
        st.pyplot(fig_roc)

    with col2:
        st.subheader("Precision–Recall Curve")
        fig_pr, ax_pr = plt.subplots()
        ax_pr.plot(recall, precision)
        ax_pr.set_xlabel("Recall")
        ax_pr.set_ylabel("Precision")
        st.pyplot(fig_pr)

    # Fraud explainer dashboard
    st.subheader("Real Fraud Examples")
    cols_to_show = list(selected_features.keys()) + ["Fraud_Label"]
    fraud_cases = df_test[df_test["Fraud_Label"] == 1].head(5)
    st.dataframe(fraud_cases[cols_to_show])

    # Model comparison
    if "LogisticOversampled" in models_dict and "LogisticRegression" in models_dict:
        st.subheader("Model Comparison")
        oversampled = models_dict["LogisticOversampled"].predict_proba(X_user)[0, 1]
        original = models_dict["LogisticRegression"].predict_proba(X_user)[0, 1]
        st.write(f"**Oversampled Model:** {oversampled:.2%}")
        st.write(f"**Original Model:** {original:.2%}")
        if oversampled >= threshold and original < threshold:
            st.warning(" Oversampled model flags this as fraud, original does not.")
        elif original >= threshold and oversampled < threshold:
            st.warning(" Original model flags this as fraud, oversampled does not.")
        else:
            st.info(" Both models agree on the prediction.")

except Exception as e:
    st.warning(f"Evaluation section skipped: {e}")