# Breast Cancer Classification Dashboard

Repository: [github.com/Likhitasuresh123/Breast-Cancer-Classification-and-Model-Comparison-Dashboard](https://github.com/Likhitasuresh123/Breast-Cancer-Classification-and-Model-Comparison-Dashboard)

## Problem statement

This project predicts whether a breast mass is malignant or benign from diagnostic measurements. It is a binary classification problem. The Streamlit dashboard compares five classifiers and evaluates them on a held-out labeled test CSV without retraining.

## Dataset description

The project uses the public [UCI Breast Cancer Wisconsin Diagnostic dataset](https://archive.ics.uci.edu/dataset/17/breast+cancer+wisconsin+diagnostic), loaded through scikit-learn's packaged copy of the UCI data. It contains 569 instances, 30 numeric input features, and the binary target column `diagnosis` with the labels `malignant` and `benign`. The included `test_data.csv` is the stratified held-out test set used by the app. The training script can reproduce the split and model artifacts.

The classes are imbalanced: 357 benign (62.7%) and 212 malignant (37.3%). The stratified 80/20 split keeps this ratio, giving 455 training rows and 114 test rows (72 benign, 42 malignant).

### Dataset limitations

- The dataset is small (569 samples), so the test set has only 114 rows. A single misclassification changes accuracy by about 0.9 percentage points, and close scores between models are not statistically meaningful.
- All results come from one fixed split (`random_state=42`). No cross-validation or repeated splits were run, so the ranking may change with a different split.
- The features are computed from digitized images of fine-needle aspirates from one institution. The models have not been validated on other populations, imaging setups, or clinical settings, and they are not intended for diagnostic use.
- The dataset has no missing values. The median imputer exists only to make the saved pipelines robust to incomplete uploaded CSVs.

## Models used

The five required models are Logistic Regression, Decision Tree, K-Nearest Neighbors, Gaussian Naive Bayes, and Support Vector Machine. All models use the same stratified 80/20 split. Numeric missing values are median-imputed and features are standardized inside each saved pipeline.

| Model | Configuration |
|---|---|
| Logistic Regression | `max_iter=3000`, `class_weight="balanced"` |
| Decision Tree | `max_depth=8`, `class_weight="balanced"` |
| KNN | `n_neighbors=11` |
| Naive Bayes | Gaussian, default settings |
| Support Vector Machine | RBF kernel, `C=2.0`, `class_weight="balanced"`, `probability=True` |

### Metric choices

- **Precision, recall, and F1 use macro averaging.** Each class gets equal weight, so the minority malignant class affects the score as much as the benign majority. Weighted averaging would let the benign class dominate.
- **AUC uses the predicted probability of the `malignant` class** as the positive score. It measures how well a model ranks malignant cases above benign ones, independent of the decision threshold.
- **MCC (Matthews correlation coefficient)** uses all four confusion-matrix cells and stays reliable under class imbalance. It is the best single summary of decision quality here.
- Accuracy is reported for completeness, but because the classes are imbalanced, a model that always predicted benign would still score 0.632.

Run `python train_models.py` to regenerate the saved models, `metrics.csv`, `model/metrics.json`, and `test_data.csv`. The values below come from `metrics.csv` and are rounded to three decimals.

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---|---:|---:|---:|---:|---:|---:|
| Logistic Regression | 0.974 | 0.995 | 0.974 | 0.969 | 0.972 | 0.943 |
| Decision Tree | 0.904 | 0.889 | 0.902 | 0.889 | 0.895 | 0.791 |
| KNN | 0.947 | 0.980 | 0.954 | 0.934 | 0.942 | 0.887 |
| Naive Bayes | 0.921 | 0.989 | 0.922 | 0.908 | 0.914 | 0.829 |
| Support Vector Machine | 0.982 | 0.995 | 0.986 | 0.976 | 0.981 | 0.963 |

Confusion matrices on the 114-row test set (72 benign, 42 malignant), taken from `model/metrics.json`:

| ML Model Name | Benign correct | Benign → malignant (FP) | Malignant → benign (FN) | Malignant correct | Total errors |
|---|---:|---:|---:|---:|---:|
| Logistic Regression | 71 | 1 | 2 | 40 | 3 |
| Decision Tree | 68 | 4 | 7 | 35 | 11 |
| KNN | 71 | 1 | 5 | 37 | 6 |
| Naive Bayes | 69 | 3 | 6 | 36 | 9 |
| Support Vector Machine | 72 | 0 | 2 | 40 | 2 |

### Observations

- **Support Vector Machine is the overall winner.** It has the best accuracy (0.982), macro precision (0.986), macro recall (0.976), macro F1 (0.981), and MCC (0.963). It makes only 2 errors, both missed malignant cases, and no false positives.
- **Logistic Regression is a close second.** It misses the same number of malignant cases (2) as the SVM and has one extra false positive. Its AUC (0.9954) is marginally higher than the SVM's (0.9950), so the two models rank cases almost equally well. The SVM wins on the final class decisions. This suggests the classes are close to linearly separable after standardization.
- **KNN** looks strong on accuracy (0.947) but misses 5 malignant cases. Its macro recall (0.934) is therefore clearly below the two leaders, which matters most in a screening context.
- **Gaussian Naive Bayes** has a high AUC (0.989) but only 0.914 F1. It ranks cases well, but its independence assumption over strongly correlated features (radius, perimeter, and area) shifts its decision threshold, leading to 9 errors.
- **Decision Tree** is the weakest model (MCC 0.791, AUC 0.889, 11 errors, including 7 missed malignant cases). A single tree with `max_depth=8` overfits the small training set. Its probabilities are coarse leaf frequencies, which explains the low AUC.
- **False negatives are the clinically important error.** Malignant cases predicted as benign would be missed. The SVM and Logistic Regression have the fewest (2 each), and the Decision Tree has the most (7).
- Because of the small test set, the SVM's lead over Logistic Regression is a single test case. Cross-validation would be needed before treating the difference as meaningful.

## Streamlit app

The app provides:

1. CSV upload for labeled test data, with a downloadable sample `test_data.csv`.
2. A dropdown for selecting any saved model, with a short configuration card.
3. A light/dark appearance switch in the sidebar.
4. An **Evaluation** tab with Accuracy, AUC, precision, recall, F1, and MCC cards (each showing the model's rank among the five), a confusion matrix, an ROC curve, and a classification report.
5. A **Predictions** tab with a filterable prediction preview, probability bars, and a predictions CSV download.
6. A **Model comparison** tab that scores all five saved models on the same uploaded data and shows a leaderboard, a metric heatmap, and a per-metric ranking chart.
7. A **Dataset** tab with class balance, a feature distribution explorer, and summary statistics.

The app includes a binary probability fallback for older serialized Logistic Regression artifacts. All dependency versions in `requirements.txt` are pinned. The saved models were serialized with scikit-learn 1.9.0 and joblib 1.6.0, so the deployed app must load them with the same versions.

## Project structure

```
app.py               Streamlit dashboard (loads saved models, never retrains)
train_models.py      Reproducible training script (split, 5 models, metrics)
requirements.txt     Pinned runtime dependencies
test_data.csv        Stratified held-out test set (114 rows) used by the app
metrics.csv          Comparison table generated by train_models.py
model/               Saved pipelines (*.joblib) and metrics.json
data/                Copy of the UCI Breast Cancer Wisconsin Diagnostic data
```

All paths in `app.py` are relative to the script's location, so the app works from any working directory and on Streamlit Community Cloud.

## Run locally

Use Python 3.11 or newer (scikit-learn 1.9.0 requires 3.11+).

```bash
pip install -r requirements.txt
streamlit run app.py
```

The trained models are committed in `model/`, so you do not need to retrain before launching the app. To regenerate the models, `test_data.csv`, `metrics.csv`, and `model/metrics.json`, run `python train_models.py`. Because the split and the models use `random_state=42`, a clean run reproduces the committed results.

## Deploy on Streamlit Community Cloud

1. Push this repository to GitHub, including `model/*.joblib` and `test_data.csv`.
2. Sign in at [share.streamlit.io](https://share.streamlit.io) with GitHub and choose **Create app**.
3. Select this repository, the `main` branch, and `app.py` as the main file path.
4. Under **Advanced settings**, choose Python 3.11 or newer (for example, 3.12).
5. Click **Deploy**. Streamlit Cloud installs `requirements.txt` automatically.

## Validation checklist

The following checks were run before deployment:

- `python -m py_compile app.py train_models.py` compiles both scripts without errors.
- A clean `python train_models.py` run reproduces the committed `model/*.joblib`, `metrics.csv`, and `test_data.csv` exactly.
- `streamlit run app.py` starts headless, and `/_stcore/health` returns `ok`.
- For each of the five models, the dashboard shows the same six metrics as `metrics.csv`, with a confusion matrix, classification report, and prediction preview.
- Uploading a copy of `test_data.csv` with `B`/`M` labels and reordered columns produces the same metrics as the bundled file. For example, the Support Vector Machine misclassifies 2 of 114 rows.
