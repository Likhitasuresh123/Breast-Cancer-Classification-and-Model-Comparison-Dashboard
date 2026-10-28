from pathlib import Path
import json

import joblib
import pandas as pd
from sklearn.compose import make_column_transformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import load_breast_cancer

ROOT = Path(__file__).resolve().parent
TARGET = "diagnosis"
RANDOM_STATE = 42

MODELS = {
    "Logistic Regression": LogisticRegression(max_iter=3000, class_weight="balanced"),
    "Decision Tree": DecisionTreeClassifier(max_depth=8, random_state=RANDOM_STATE, class_weight="balanced"),
    "KNN": KNeighborsClassifier(n_neighbors=11),
    "Naive Bayes": GaussianNB(),
    "Support Vector Machine": SVC(C=2.0, kernel="rbf", probability=True, class_weight="balanced", random_state=RANDOM_STATE),
}


def evaluate(model, features, target):
    predictions = model.predict(features)
    probabilities = model.predict_proba(features)
    labels = model.classes_
    return {
        "Accuracy": accuracy_score(target, predictions),
        "AUC": roc_auc_score(target == "malignant", probabilities[:, list(labels).index("malignant")]),
        "Precision": precision_score(target, predictions, average="macro", zero_division=0),
        "Recall": recall_score(target, predictions, average="macro", zero_division=0),
        "F1": f1_score(target, predictions, average="macro", zero_division=0),
        "MCC": matthews_corrcoef(target, predictions),
        "confusion_matrix": confusion_matrix(target, predictions, labels=labels).tolist(),
        "labels": [str(label) for label in labels],
    }


def main():
    dataset = load_breast_cancer(as_frame=True)
    data = dataset.frame.copy()
    data.columns = [column.replace(" ", "_") for column in data.columns]
    data = data.rename(columns={"target": TARGET})
    data[TARGET] = data[TARGET].map({0: "malignant", 1: "benign"})
    features = data.drop(columns=[TARGET])
    target = data[TARGET]
    x_train, x_test, y_train, y_test = train_test_split(
        features,
        target,
        test_size=0.2,
        random_state=RANDOM_STATE,
        stratify=target,
    )

    output_dir = ROOT / "model"
    output_dir.mkdir(exist_ok=True)
    test_data = x_test.copy()
    test_data[TARGET] = y_test
    test_data.to_csv(ROOT / "test_data.csv", index=False)

    metrics = {}
    for name, estimator in MODELS.items():
        model = make_pipeline(
            SimpleImputer(strategy="median"),
            StandardScaler(),
            estimator,
        )
        model.fit(x_train, y_train)
        metrics[name] = evaluate(model, x_test, y_test)
        joblib.dump(model, output_dir / f"{name.lower().replace(' ', '_')}.joblib")

    with (output_dir / "metrics.json").open("w", encoding="utf-8") as file:
        json.dump(metrics, file, indent=2)

    comparison = pd.DataFrame(metrics).T.drop(columns=["confusion_matrix", "labels"])
    comparison.to_csv(ROOT / "metrics.csv")
    print(comparison.round(4).to_string())
    print(f"\nSaved models and test data in {ROOT}")


if __name__ == "__main__":
    main()
