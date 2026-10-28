# Breast Cancer Classification Dashboard

## Problem statement

This project predicts whether a breast mass is malignant or benign from diagnostic measurements. It is a binary classification problem. The Streamlit dashboard compares five classifiers and evaluates them on a held-out labeled test CSV without retraining.

## Dataset description

The project uses the public [UCI Breast Cancer Wisconsin Diagnostic dataset](https://archive.ics.uci.edu/dataset/17/breast+cancer+wisconsin+diagnostic), loaded through scikit-learn's packaged copy of the UCI data. It contains 569 instances, 30 numeric input features, and the binary target column `diagnosis` with the labels `malignant` and `benign`. The included `test_data.csv` is the stratified held-out test set used by the app. The training script can reproduce the split and model artifacts.


## Models used

The five required models are Logistic Regression, Decision Tree, K-Nearest Neighbors, Gaussian Naive Bayes, and Support Vector Machine. All models use the same stratified 80/20 split. Numeric missing values are median-imputed and features are standardized inside each saved pipeline. Precision, recall, and F1 use macro averaging; AUC uses the probability of the malignant class.

Run `python train_models.py` to regenerate the saved models and the comparison table.

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---|---:|---:|---:|---:|---:|---:|
| Logistic Regression | 0.974 | 0.995 | 0.974 | 0.969 | 0.972 | 0.943 |
| Decision Tree | 0.904 | 0.889 | 0.902 | 0.889 | 0.895 | 0.791 |
| KNN | 0.947 | 0.980 | 0.954 | 0.934 | 0.942 | 0.887 |
| Naive Bayes | 0.921 | 0.989 | 0.922 | 0.908 | 0.914 | 0.829 |
| Support Vector Machine | 0.982 | 0.995 | 0.986 | 0.976 | 0.981 | 0.963 |

### Observations

- Logistic Regression is a strong linear baseline, with 0.974 accuracy and 0.972 macro F1.
- Decision Tree is the weakest model in this split, suggesting that the selected depth does not represent the feature boundaries as effectively as the scaled models.
- KNN performs well after standardization, but trails the two strongest models on every main metric except it remains competitive on AUC.
- Gaussian Naive Bayes has high AUC but lower recall and F1, showing that probability ranking is stronger than its final class decisions.
- Support Vector Machine is the overall winner by accuracy, macro F1, MCC, and precision. Logistic Regression has the marginally higher AUC, so the winner depends on the chosen priority; this report uses the broadest classification performance as the primary criterion.

## Streamlit app

The app provides:

1. CSV upload for labeled test data.
2. A dropdown for selecting any saved model.
3. Accuracy, AUC, precision, recall, F1, and MCC display.
4. A confusion matrix and classification report.
5. A prediction preview with confidence values.

The app includes a binary probability fallback for older serialized Logistic Regression artifacts. Dependency versions are pinned so newly trained artifacts and Streamlit Cloud use the same scikit-learn and joblib versions.

Run locally:

```bash
pip install -r requirements.txt
python train_models.py
streamlit run app.py
```

For Streamlit Community Cloud, deploy `app.py` from the repository root and keep `model/`, `test_data.csv`, and `requirements.txt` in the same project.

