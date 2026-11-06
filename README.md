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
