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
CLASS_LABELS = ["benign", "malignant"]
# Accept common encodings; numeric codes follow sklearn's load_breast_cancer (0=malignant, 1=benign).
LABEL_ALIASES = {
    "b": "benign",
    "benign": "benign",
    "1": "benign",
    "m": "malignant",
    "malignant": "malignant",
    "0": "malignant",
}


@st.cache_resource
def load_model(file_name):
    return joblib.load(MODEL_DIR / file_name)


def normalize_target(series):
    return series.astype(str).str.strip().str.lower().map(LABEL_ALIASES)


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
    st.markdown(
        "The CSV must contain the 30 diagnostic measurements and the `diagnosis` target "
        "(`benign`/`malignant`, `B`/`M`, or `1`/`0`)."
    )
    sample_path = ROOT / "test_data.csv"
    st.download_button(
        "Download sample test_data.csv",
        data=sample_path.read_bytes(),
        file_name="test_data.csv",
        mime="text/csv",
    )

if uploaded_file is None:
    data = pd.read_csv(ROOT / "test_data.csv")
    st.info("Showing the included held-out test data. Upload another compatible CSV to evaluate it.")
else:
    try:
        data = pd.read_csv(uploaded_file)
    except (pd.errors.ParserError, pd.errors.EmptyDataError, UnicodeDecodeError) as error:
        st.error(f"Could not read the uploaded CSV: {error}")
        st.stop()
    st.success(f"Loaded `{uploaded_file.name}` with {len(data)} rows.")

data.columns = [str(column).strip() for column in data.columns]
if TARGET not in data.columns:
    st.error(f"The uploaded CSV is missing the `{TARGET}` target column.")
    st.stop()

target = normalize_target(data[TARGET])
if target.isna().any():
    bad_values = sorted(data.loc[target.isna(), TARGET].astype(str).unique())[:10]
    st.error(f"Unrecognised `{TARGET}` values: {', '.join(bad_values)}")
    st.stop()

model = load_model(MODEL_FILES[model_name])
features = data.drop(columns=[TARGET])
expected_features = list(model.feature_names_in_)
missing = sorted(set(expected_features) - set(features.columns))
extra = sorted(set(features.columns) - set(expected_features))

if missing:
    st.error(f"Missing required feature columns: {', '.join(missing)}")
    st.stop()

features = features[expected_features].apply(pd.to_numeric, errors="coerce")
non_numeric = [column for column in expected_features if features[column].isna().all()]
if non_numeric:
    st.error(f"Feature columns contain no numeric values: {', '.join(non_numeric)}")
    st.stop()

predictions = model.predict(features)
probabilities = predict_probabilities(model, features)
positive_class_index = list(model.classes_).index("malignant")
malignant_probability = probabilities[:, positive_class_index]
has_both_classes = target.nunique() == 2
metrics = {
    "Accuracy": accuracy_score(target, predictions),
    "AUC": roc_auc_score(target == "malignant", malignant_probability) if has_both_classes else float("nan"),
    "Precision": precision_score(target, predictions, average="macro", zero_division=0),
    "Recall": recall_score(target, predictions, average="macro", zero_division=0),
    "F1": f1_score(target, predictions, average="macro", zero_division=0),
    "MCC": matthews_corrcoef(target, predictions),
}

st.subheader(f"{model_name} results")
st.caption(f"Evaluated {len(data)} rows with the saved pipeline (no retraining). Precision, recall and F1 are macro-averaged.")
metric_columns = st.columns(6)
for column, metric in zip(metric_columns, ["Accuracy", "AUC", "Precision", "Recall", "F1", "MCC"]):
    value = metrics[metric]
    column.metric(metric, "n/a" if np.isnan(value) else f"{value:.3f}")
if not has_both_classes:
    st.warning("AUC needs both classes in the uploaded data, so it is not reported.")

left, right = st.columns([1.1, 1])
with left:
    st.subheader("Confusion matrix")
    figure, axis = plt.subplots(figsize=(6, 4))
    ConfusionMatrixDisplay.from_predictions(
        target, predictions, labels=CLASS_LABELS, ax=axis, colorbar=False, cmap="Blues"
    )
    axis.set_xlabel("Predicted diagnosis")
    axis.set_ylabel("Actual diagnosis")
    st.pyplot(figure, clear_figure=True)

with right:
    st.subheader("Classification report")
    report = classification_report(
        target, predictions, labels=CLASS_LABELS, output_dict=True, zero_division=0
    )
    st.dataframe(pd.DataFrame(report).T.round(3), width="stretch")

st.subheader("Prediction preview")
preview = pd.DataFrame(
    {
        "actual_diagnosis": target.values,
        "predicted_diagnosis": predictions,
        "malignant_probability": malignant_probability.round(3),
        "prediction_confidence": probabilities.max(axis=1).round(3),
    },
    index=data.index,
)
preview["correct"] = preview["actual_diagnosis"] == preview["predicted_diagnosis"]
preview = pd.concat([preview, features], axis=1)

preview_controls = st.columns([1, 2])
only_errors = preview_controls[0].checkbox("Show misclassified rows only")
row_limit = preview_controls[1].number_input(
    "Rows to show", min_value=1, max_value=max(1, len(preview)), value=min(20, max(1, len(preview)))
)
shown = preview[~preview["correct"]] if only_errors else preview
st.caption(f"{int((~preview['correct']).sum())} of {len(preview)} rows misclassified.")
st.dataframe(shown.head(row_limit), width="stretch")
st.download_button(
    "Download predictions CSV",
    data=preview.to_csv(index=False).encode("utf-8"),
    file_name=f"{MODEL_FILES[model_name].removesuffix('.joblib')}_predictions.csv",
    mime="text/csv",
)
if extra:
    st.caption(f"Ignored extra columns: {', '.join(extra)}")
