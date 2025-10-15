# streamlit_app.py

import streamlit as st
import pandas as pd
import numpy as np
import joblib
from pathlib import Path
import matplotlib.pyplot as plt
from sklearn.metrics import (
    confusion_matrix,
    roc_curve,
    roc_auc_score,
    precision_recall_curve
)

# 1. Page config
st.set_page_config(page_title="Fraud Model Demo", layout="wide")

# 2. Load pickled models and feature names
@st.cache(allow_output_mutation=True)
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

# 3. Sidebar — model selector
if models_dict:
    model_name = st.sidebar.selectbox("Select model", list(models_dict.keys()))
    model = models_dict[model_name]
else:
    st.error("No models found in /models folder.")
    st.stop()

# 4. Sidebar — dynamic feature inputs
st.sidebar.header("Input Features")
user_inputs = {}
for feat in feature_names:
    user_inputs[feat] = st.sidebar.number_input(label=feat, value=0.0)

# 5. Sidebar — decision threshold
threshold = st.sidebar.slider("Decision threshold", 0.0, 1.0, 0.5)

# 6. Display user inputs
X_user = pd.DataFrame([user_inputs])
st.write("## User Inputs")
st.dataframe(X_user)

# 7. Run prediction
proba = model.predict_proba(X_user)[0, 1]
label = "Fraud" if proba >= threshold else "Legit"

st.write("## Prediction")
st.metric("Fraud Probability", f"{proba:.2%}")
st.metric("Predicted Class", label)

# 8. Evaluation on held-out test set
@st.cache(allow_output_mutation=True)
def load_test_set(path="models/test_set.csv"):
    df = pd.read_csv(path)
    X_test = df[feature_names]
    y_test = df["Fraud_Label"]
    probs = model.predict_proba(X_test)[:, 1]
    return y_test, probs

try:
    y_test, probs = load_test_set()
    y_pred = (probs >= threshold).astype(int)

    st.write("### Confusion Matrix")
    cm = confusion_matrix(y_test, y_pred)
    fig_cm, ax_cm = plt.subplots()
    ax_cm.matshow(cm, cmap="Blues")
    for (i, j), v in np.ndenumerate(cm):
        ax_cm.text(j, i, str(v), ha="center", va="center")
    st.pyplot(fig_cm)

    fpr, tpr, _ = roc_curve(y_test, probs)
    precision, recall, _ = precision_recall_curve(y_test, probs)
    auc_score = roc_auc_score(y_test, probs)

    col1, col2 = st.columns(2)
    with col1:
        st.write("### ROC Curve")
        fig_roc, ax_roc = plt.subplots()
        ax_roc.plot(fpr, tpr, label=f"AUC = {auc_score:.3f}")
        ax_roc.plot([0, 1], [0, 1], "--", color="gray")
        ax_roc.set_xlabel("False Positive Rate")
        ax_roc.set_ylabel("True Positive Rate")
        ax_roc.legend()
        st.pyplot(fig_roc)

    with col2:
        st.write("### Precision–Recall Curve")
        fig_pr, ax_pr = plt.subplots()
        ax_pr.plot(recall, precision)
        ax_pr.set_xlabel("Recall")
        ax_pr.set_ylabel("Precision")
        st.pyplot(fig_pr)

except Exception as e:
    st.warning(f"Evaluation section skipped: {e}")