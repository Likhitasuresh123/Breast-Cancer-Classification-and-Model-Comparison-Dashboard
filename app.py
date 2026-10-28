from pathlib import Path
import numpy as np

import joblib
import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)

ROOT = Path(__file__).resolve().parent
TARGET = "diagnosis"
MODEL_DIR = ROOT / "model"
MODEL_FILES = {
    "Logistic Regression": "logistic_regression.joblib",
    "Decision Tree": "decision_tree.joblib",
    "KNN": "knn.joblib",
    "Naive Bayes": "naive_bayes.joblib",
    "Support Vector Machine": "support_vector_machine.joblib",
}


def predict_probabilities(model, features):
    try:
        return model.predict_proba(features)
    except AttributeError as error:
        if len(model.classes_) != 2 or not hasattr(model, "decision_function"):
            raise error

        scores = np.asarray(model.decision_function(features), dtype=float)
        second_class_probability = 1 / (1 + np.exp(-np.clip(scores, -40, 40)))
        probabilities = np.empty((len(features), 2), dtype=float)
        probabilities[:, 1] = second_class_probability
        probabilities[:, 0] = 1 - second_class_probability
        return probabilities

st.set_page_config(page_title="Breast Cancer Classifier", page_icon="🔬", layout="wide")
st.title("Breast Cancer Classification Lab")
st.caption("Compare five saved classifiers on a labeled Wisconsin Diagnostic test set.")

with st.sidebar:
    st.header("Evaluation controls")
    model_name = st.selectbox("Choose a model", list(MODEL_FILES))
    uploaded_file = st.file_uploader("Upload labeled test data", type="csv")
    st.markdown("The CSV must contain the 30 diagnostic measurements and the `diagnosis` target.")

if uploaded_file is None:
    data = pd.read_csv(ROOT / "test_data.csv")
    st.info("Showing the included held-out test data. Upload another compatible CSV to evaluate it.")
else:
    data = pd.read_csv(uploaded_file)

if TARGET not in data.columns:
    st.error(f"The uploaded CSV is missing the `{TARGET}` target column.")
    st.stop()

model = joblib.load(MODEL_DIR / MODEL_FILES[model_name])
features = data.drop(columns=[TARGET])
target = data[TARGET]
expected_features = list(model.feature_names_in_)
missing = sorted(set(expected_features) - set(features.columns))
extra = sorted(set(features.columns) - set(expected_features))

if missing:
    st.error(f"Missing required feature columns: {', '.join(missing)}")
    st.stop()

features = features[expected_features]
predictions = model.predict(features)
probabilities = predict_probabilities(model, features)
positive_class_index = list(model.classes_).index("malignant")
metrics = {
    "Accuracy": accuracy_score(target, predictions),
    "AUC": roc_auc_score(
        target == "malignant", probabilities[:, positive_class_index]
    ),
    "Precision": precision_score(target, predictions, average="macro", zero_division=0),
    "Recall": recall_score(target, predictions, average="macro", zero_division=0),
    "F1": f1_score(target, predictions, average="macro", zero_division=0),
    "MCC": matthews_corrcoef(target, predictions),
}

st.subheader(f"{model_name} results")
metric_columns = st.columns(6)
for column, metric in zip(metric_columns, ["Accuracy", "AUC", "Precision", "Recall", "F1", "MCC"]):
    column.metric(metric, f"{metrics[metric]:.3f}")

left, right = st.columns([1.1, 1])
with left:
    st.subheader("Confusion matrix")
    figure, axis = plt.subplots(figsize=(6, 4))
    ConfusionMatrixDisplay.from_predictions(target, predictions, ax=axis, colorbar=False)
    axis.set_xlabel("Predicted diagnosis")
    axis.set_ylabel("Actual diagnosis")
    st.pyplot(figure, clear_figure=True)

with right:
    st.subheader("Classification report")
    report = classification_report(target, predictions, output_dict=True, zero_division=0)
    st.dataframe(pd.DataFrame(report).T.round(3), use_container_width=True)

st.subheader("Prediction preview")
preview = data.copy()
preview["predicted_diagnosis"] = predictions
preview["prediction_confidence"] = probabilities.max(axis=1).round(3)
st.dataframe(preview.head(20), use_container_width=True)
if extra:
    st.caption(f"Ignored extra columns: {', '.join(extra)}")
