from pathlib import Path
from string import Template

import joblib
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, Normalize, to_rgb
import numpy as np
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
    roc_curve,
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
MODEL_NOTES = {
    "Logistic Regression": ("Linear", "max_iter=3000, balanced class weights"),
    "Decision Tree": ("Tree", "max_depth=8, balanced class weights"),
    "KNN": ("Instance-based", "k = 11 neighbours on standardized features"),
    "Naive Bayes": ("Probabilistic", "Gaussian likelihoods, default settings"),
    "Support Vector Machine": ("Kernel", "RBF kernel, C=2.0, balanced class weights"),
}
METRIC_NAMES = ["Accuracy", "AUC", "Precision", "Recall", "F1", "MCC"]
METRIC_HELP = {
    "Accuracy": "Share of correct predictions",
    "AUC": "Ranking quality of malignant probability",
    "Precision": "Macro-averaged precision",
    "Recall": "Macro-averaged recall",
    "F1": "Macro-averaged F1 score",
    "MCC": "Matthews correlation coefficient",
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

PALETTES = {
    "Light": {
        "bg": "#f4f6fb",
        "surface": "#ffffff",
        "sidebar": "#ffffff",
        "text": "#1c2433",
        "muted": "#5f6b80",
        "border": "#e2e7f0",
        "accent": "#d6336c",
        "accent2": "#7048e8",
        "benign": "#20a387",
        "malignant": "#d6336c",
        "shadow": "0 6px 18px rgba(28, 36, 51, 0.06)",
        "cmap_colors": ["#fff0f6", "#f06595", "#862e9c"],
    },
    "Dark": {
        "bg": "#0e1320",
        "surface": "#171f31",
        "sidebar": "#121929",
        "text": "#e7eaf2",
        "muted": "#9aa5bb",
        "border": "#283249",
        "accent": "#f06595",
        "accent2": "#9775fa",
        "benign": "#38d9a9",
        "malignant": "#f06595",
        "shadow": "0 6px 18px rgba(0, 0, 0, 0.35)",
        "cmap_colors": ["#1f2a42", "#9c36b5", "#f783ac"],
    },
}

CSS = Template(
    """
<style>
.stApp { background: $bg; color: $text; }
[data-testid="stHeader"] { background: transparent; }
[data-testid="stSidebar"] { background: $sidebar; border-right: 1px solid $border; }
.block-container { padding-top: 2.2rem; max-width: 1320px; }
.stApp h1, .stApp h2, .stApp h3, .stApp h4, .stApp p, .stApp li, .stApp label,
[data-testid="stMarkdownContainer"], [data-testid="stWidgetLabel"] { color: $text; }
[data-testid="stCaptionContainer"], [data-testid="stCaptionContainer"] p { color: $muted; }
[data-baseweb="select"] > div, [data-baseweb="input"], [data-baseweb="input"] input,
[data-baseweb="base-input"], [data-testid="stNumberInput"] input,
[data-testid="stSelectbox"] [role="group"] {
  background: $surface !important; color: $text !important; border-color: $border !important;
}
[data-testid="stSelectbox"] input, [data-testid="stSelectbox"] button, [data-testid="stSelectbox"] svg { color: $text !important; }
[role="listbox"] { background: $surface !important; border: 1px solid $border; }
[role="option"] { background: $surface !important; color: $text !important; }
[role="option"]:hover, [role="option"][aria-selected="true"] { background: $border !important; }
[data-testid="stSidebar"] code, [data-testid="stCaptionContainer"] code { background: $border; color: $accent; }
[data-baseweb="select"] svg, [data-testid="stNumberInput"] button { color: $text; }
[data-testid="stNumberInput"] button { background: $surface; border-color: $border; }
[data-baseweb="popover"] ul, [data-baseweb="popover"] li { background: $surface; color: $text; }
[data-baseweb="popover"] li:hover { background: $border; }
[data-testid="stFileUploaderDropzone"] { background: $surface; border: 1.5px dashed $accent; border-radius: 12px; }
[data-testid="stFileUploaderDropzone"] *, [data-testid="stFileUploaderFile"] * { color: $text; }
[data-testid="stFileUploaderDropzone"] button { background: $bg; border: 1px solid $border; }
[data-testid="stDownloadButton"] button, .stButton button {
  background: transparent; color: $accent; border: 1px solid $accent; border-radius: 10px; font-weight: 600;
}
[data-testid="stDownloadButton"] button:hover, .stButton button:hover { background: $accent; color: #ffffff; }
[data-testid="stDownloadButton"] button:hover p, .stButton button:hover p { color: #ffffff; }
[data-baseweb="tab-list"] { gap: 0.4rem; border-bottom: 1px solid $border; }
[data-baseweb="tab"] { background: transparent; border-radius: 10px 10px 0 0; padding: 0.55rem 1rem; }
[data-baseweb="tab"] p { color: $muted; font-weight: 600; }
[data-baseweb="tab"][aria-selected="true"] { background: $surface; }
[data-baseweb="tab"][aria-selected="true"] p { color: $accent; }
[data-baseweb="tab-highlight"] { background: $accent; }
[data-baseweb="tab-border"] { display: none; }
[data-testid="stAlert"] { border-radius: 12px; }
[data-testid="stExpander"] details { background: $surface; border: 1px solid $border; border-radius: 12px; }
[class*="st-key-card"] {
  background: $surface; border: 1px solid $border !important; border-radius: 16px; box-shadow: $shadow; padding: 0.4rem 0.6rem;
}
.hero {
  background: linear-gradient(120deg, $accent2 0%, $accent 100%);
  border-radius: 20px; padding: 1.6rem 1.9rem; margin-bottom: 1.2rem; box-shadow: $shadow;
}
.hero h1 { color: #ffffff !important; margin: 0; padding: 0; font-size: 2.05rem; }
.hero p { color: rgba(255, 255, 255, 0.9) !important; margin: 0.35rem 0 0.9rem 0; font-size: 1.02rem; }
.pill {
  display: inline-block; padding: 0.22rem 0.75rem; margin: 0 0.4rem 0.3rem 0; border-radius: 999px;
  background: rgba(255, 255, 255, 0.18); color: #ffffff; font-size: 0.8rem; font-weight: 600;
  border: 1px solid rgba(255, 255, 255, 0.35);
}
.section-title { font-size: 1.15rem; font-weight: 700; color: $text; margin: 0.4rem 0 0.6rem 0; }
.kpi-grid { display: grid; grid-template-columns: repeat(6, minmax(0, 1fr)); gap: 0.8rem; margin: 0.4rem 0 1.1rem 0; }
@media (max-width: 1000px) { .kpi-grid { grid-template-columns: repeat(3, minmax(0, 1fr)); } }
@media (max-width: 600px) { .kpi-grid { grid-template-columns: repeat(2, minmax(0, 1fr)); } }
.kpi { background: $surface; border: 1px solid $border; border-radius: 14px; padding: 0.85rem 1rem; box-shadow: $shadow; }
.kpi .label { font-size: 0.74rem; letter-spacing: 0.07em; text-transform: uppercase; color: $muted; font-weight: 700; }
.kpi .value { font-size: 1.75rem; font-weight: 800; color: $text; line-height: 1.25; }
.kpi .bar { height: 6px; background: $border; border-radius: 3px; overflow: hidden; margin: 0.35rem 0; }
.kpi .bar span { display: block; height: 100%; background: linear-gradient(90deg, $accent2, $accent); }
.kpi .rank { font-size: 0.74rem; color: $muted; }
.kpi .rank b { color: $accent; }
.model-card { background: $surface; border: 1px solid $border; border-radius: 14px; padding: 0.8rem 0.95rem; }
.model-card .tag { font-size: 0.72rem; font-weight: 700; color: $accent; text-transform: uppercase; letter-spacing: 0.06em; }
.model-card .name { font-weight: 700; color: $text; margin: 0.15rem 0; }
.model-card .cfg { font-size: 0.82rem; color: $muted; }
.footer { text-align: center; color: $muted; font-size: 0.8rem; margin-top: 2rem; }
.empty-state {
  background: $surface; border: 1.5px dashed $accent; border-radius: 18px; padding: 2.2rem 2rem;
  text-align: center; box-shadow: $shadow;
}
.empty-state .icon { font-size: 2.6rem; }
.empty-state h3 { color: $text; margin: 0.4rem 0; }
.empty-state p { color: $muted !important; }
.empty-state .steps { display: grid; grid-template-columns: repeat(3, minmax(0, 1fr)); gap: 0.8rem; margin-top: 1.2rem; text-align: left; }
@media (max-width: 800px) { .empty-state .steps { grid-template-columns: 1fr; } }
.empty-state .steps div { background: $bg; border: 1px solid $border; border-radius: 12px; padding: 0.8rem 0.9rem; color: $text; font-size: 0.9rem; }
.empty-state .steps b {
  display: inline-flex; align-items: center; justify-content: center; width: 1.6rem; height: 1.6rem; margin-right: 0.5rem;
  border-radius: 50%; background: linear-gradient(120deg, $accent2, $accent); color: #ffffff;
}
.empty-state code { background: $border; color: $accent; }
$dataframe_css
</style>
"""
)


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


def evaluate(model, features, target):
    predictions = model.predict(features)
    probabilities = predict_probabilities(model, features)
    malignant_probability = probabilities[:, list(model.classes_).index("malignant")]
    has_both_classes = target.nunique() == 2
    metrics = {
        "Accuracy": accuracy_score(target, predictions),
        "AUC": roc_auc_score(target == "malignant", malignant_probability) if has_both_classes else float("nan"),
        "Precision": precision_score(target, predictions, average="macro", zero_division=0),
        "Recall": recall_score(target, predictions, average="macro", zero_division=0),
        "F1": f1_score(target, predictions, average="macro", zero_division=0),
        "MCC": matthews_corrcoef(target, predictions),
    }
    return metrics, predictions, probabilities, malignant_probability


def styled_figure(palette, size=(6, 4.2)):
    figure, axis = plt.subplots(figsize=size)
    figure.patch.set_facecolor(palette["surface"])
    axis.set_facecolor(palette["surface"])
    for spine in axis.spines.values():
        spine.set_color(palette["border"])
    axis.tick_params(colors=palette["muted"])
    axis.xaxis.label.set_color(palette["text"])
    axis.yaxis.label.set_color(palette["text"])
    axis.title.set_color(palette["text"])
    return figure, axis


def section(title):
    st.markdown(f'<div class="section-title">{title}</div>', unsafe_allow_html=True)


def kpi_cards(metrics, comparison, model_name):
    cards = []
    for metric in METRIC_NAMES:
        value = metrics[metric]
        if np.isnan(value):
            display, width, rank_text = "n/a", 0, "needs both classes"
        else:
            display = f"{value:.3f}"
            width = max(0.0, min(1.0, value)) * 100
            rank = int(comparison[metric].rank(ascending=False, method="min")[model_name])
            rank_text = f"Rank <b>#{rank}</b> of {len(comparison)}"
        cards.append(
            f'<div class="kpi" title="{METRIC_HELP[metric]}"><div class="label">{metric}</div>'
            f'<div class="value">{display}</div><div class="bar"><span style="width:{width:.1f}%"></span></div>'
            f'<div class="rank">{rank_text}</div></div>'
        )
    st.markdown(f'<div class="kpi-grid">{"".join(cards)}</div>', unsafe_allow_html=True)


st.set_page_config(page_title="Breast Cancer Classifier", page_icon="🔬", layout="wide")

base_theme = getattr(st.context.theme, "type", None) or "light"

with st.sidebar:
    st.markdown("## 🔬 Diagnostic Lab")
    mode = st.radio(
        "Appearance",
        ["Light", "Dark"],
        index=1 if base_theme == "dark" else 0,
        horizontal=True,
        format_func=lambda option: f"☀️ {option}" if option == "Light" else f"🌙 {option}",
    )
    st.divider()
    model_name = st.selectbox("Choose a model", list(MODEL_FILES))
    family, configuration = MODEL_NOTES[model_name]
    st.markdown(
        f'<div class="model-card"><div class="tag">{family}</div><div class="name">{model_name}</div>'
        f'<div class="cfg">{configuration}</div></div>',
        unsafe_allow_html=True,
    )
    st.write("")
    uploaded_file = st.file_uploader("Upload labeled test data", type="csv")
    st.caption(
        "The CSV needs the 30 diagnostic measurements and a `diagnosis` column "
        "(`benign`/`malignant`, `B`/`M`, or `1`/`0`)."
    )
    st.download_button(
        "⬇️ Download sample test_data.csv",
        data=(ROOT / "test_data.csv").read_bytes(),
        file_name="test_data.csv",
        mime="text/csv",
        width="stretch",
    )

palette = PALETTES[mode]
colormap = LinearSegmentedColormap.from_list(f"{mode.lower()}_diagnostic", palette["cmap_colors"])
# Streamlit draws dataframes on a canvas that ignores page CSS, so invert them when the
# chosen appearance differs from the active Streamlit base theme.
dataframe_css = (
    '[data-testid="stDataFrame"] { filter: invert(0.93) hue-rotate(180deg); }'
    if mode.lower() != base_theme
    else ""
)
st.markdown(CSS.substitute(palette, dataframe_css=dataframe_css), unsafe_allow_html=True)

st.markdown(
    '<div class="hero"><h1>Breast Cancer Classification Lab</h1>'
    "<p>Evaluate and compare five saved classifiers on labeled Wisconsin Diagnostic data, with no retraining.</p>"
    '<span class="pill">5 models</span><span class="pill">6 metrics</span>'
    '<span class="pill">UCI Wisconsin Diagnostic</span><span class="pill">Saved pipelines</span></div>',
    unsafe_allow_html=True,
)

if uploaded_file is None:
    st.markdown(
        '<div class="empty-state"><div class="icon">📂</div>'
        "<h3>Upload labeled test data to begin</h3>"
        "<p>No metrics are shown until you upload a CSV. Use <b>Upload labeled test data</b> in the sidebar.</p>"
        '<div class="steps">'
        "<div><b>1</b>Choose a model in the sidebar</div>"
        "<div><b>2</b>Upload a CSV with the 30 features and a <code>diagnosis</code> column</div>"
        "<div><b>3</b>Explore metrics, predictions, and model comparisons</div>"
        "</div></div>",
        unsafe_allow_html=True,
    )
    st.stop()

try:
    data = pd.read_csv(uploaded_file)
except (pd.errors.ParserError, pd.errors.EmptyDataError, UnicodeDecodeError) as error:
    st.error(f"Could not read the uploaded CSV: {error}")
    st.stop()
st.success(f"Loaded `{uploaded_file.name}` with {len(data)} rows.", icon="✅")

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

all_results = {name: evaluate(load_model(file), features, target) for name, file in MODEL_FILES.items()}
comparison = pd.DataFrame({name: result[0] for name, result in all_results.items()}).T[METRIC_NAMES]
metrics, predictions, probabilities, malignant_probability = all_results[model_name]
has_both_classes = target.nunique() == 2

evaluation_tab, predictions_tab, comparison_tab, dataset_tab = st.tabs(
    ["📊 Evaluation", "🔎 Predictions", "🏆 Model comparison", "🧬 Dataset"]
)

with evaluation_tab:
    section(f"{model_name} results")
    st.caption(f"Evaluated {len(data)} rows with the saved pipeline (no retraining). Precision, recall and F1 are macro-averaged.")
    kpi_cards(metrics, comparison, model_name)
    if not has_both_classes:
        st.warning("AUC and the ROC curve need both classes in the uploaded data, so they are not reported.")

    left, right = st.columns(2)
    with left, st.container(key="card_confusion"):
        section("Confusion matrix")
        figure, axis = styled_figure(palette)
        ConfusionMatrixDisplay.from_predictions(
            target, predictions, labels=CLASS_LABELS, ax=axis, colorbar=False, cmap=colormap
        )
        axis.set_xlabel("Predicted diagnosis")
        axis.set_ylabel("Actual diagnosis")
        st.pyplot(figure, clear_figure=True)

    with right, st.container(key="card_roc"):
        section("ROC curve")
        figure, axis = styled_figure(palette)
        if has_both_classes:
            false_positive_rate, true_positive_rate, _ = roc_curve(target == "malignant", malignant_probability)
            axis.plot(false_positive_rate, true_positive_rate, color=palette["accent"], linewidth=2.4,
                      label=f"{model_name} (AUC = {metrics['AUC']:.3f})")
            axis.fill_between(false_positive_rate, true_positive_rate, color=palette["accent"], alpha=0.12)
            axis.plot([0, 1], [0, 1], linestyle="--", color=palette["muted"], linewidth=1, label="Random guess")
            legend = axis.legend(loc="lower right", frameon=False)
            for text in legend.get_texts():
                text.set_color(palette["text"])
        else:
            axis.text(0.5, 0.5, "Needs both classes", ha="center", va="center", color=palette["muted"])
        axis.set_xlabel("False positive rate")
        axis.set_ylabel("True positive rate (malignant)")
        st.pyplot(figure, clear_figure=True)

    with st.container(key="card_report"):
        section("Classification report")
        report = classification_report(
            target, predictions, labels=CLASS_LABELS, output_dict=True, zero_division=0
        )
        st.dataframe(pd.DataFrame(report).T.round(3), width="stretch")

with predictions_tab:
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
    error_count = int((~preview["correct"]).sum())

    summary = st.columns(3)
    summary[0].metric("Rows evaluated", len(preview))
    summary[1].metric("Correct predictions", len(preview) - error_count)
    summary[2].metric("Misclassified", error_count)

    preview_controls = st.columns([1, 2])
    only_errors = preview_controls[0].checkbox("Show misclassified rows only")
    row_limit = preview_controls[1].number_input(
        "Rows to show", min_value=1, max_value=max(1, len(preview)), value=min(20, max(1, len(preview)))
    )
    shown = preview[~preview["correct"]] if only_errors else preview
    st.caption(f"{error_count} of {len(preview)} rows misclassified.")
    st.dataframe(
        shown.head(row_limit),
        width="stretch",
        column_config={
            "malignant_probability": st.column_config.ProgressColumn(
                "Malignant probability", min_value=0.0, max_value=1.0, format="%.3f"
            ),
            "prediction_confidence": st.column_config.ProgressColumn(
                "Confidence", min_value=0.0, max_value=1.0, format="%.3f"
            ),
            "correct": st.column_config.CheckboxColumn("Correct"),
        },
    )
    st.download_button(
        "⬇️ Download predictions CSV",
        data=preview.to_csv(index=False).encode("utf-8"),
        file_name=f"{MODEL_FILES[model_name].removesuffix('.joblib')}_predictions.csv",
        mime="text/csv",
    )
    if extra:
        st.caption(f"Ignored extra columns: {', '.join(extra)}")

with comparison_tab:
    section("All five models on the current data")
    leader = comparison["MCC"].idxmax()
    st.caption(f"Best MCC on this data: **{leader}** ({comparison.loc[leader, 'MCC']:.3f}). Highlighted cells mark the best value per metric.")
    st.dataframe(
        comparison.style.format("{:.3f}", na_rep="n/a").highlight_max(axis=0, color="#f8c8dc"),
        width="stretch",
    )

    heatmap_column, ranking_column = st.columns([1.25, 1])
    with heatmap_column, st.container(key="card_heatmap"):
        section("Metric heatmap")
        figure, axis = styled_figure(palette, size=(5.2, 3.4))
        values = comparison.to_numpy(dtype=float)
        norm = Normalize(vmin=max(0.0, np.nanmin(values) - 0.05), vmax=1.0)
        filled = np.where(np.isnan(values), norm.vmin, values)
        axis.imshow(filled, cmap=colormap, norm=norm, aspect="auto")
        axis.set_xticks(range(len(METRIC_NAMES)), METRIC_NAMES)
        axis.set_yticks(range(len(comparison)), [name.replace("Support Vector Machine", "SVM") for name in comparison.index])
        axis.tick_params(length=0)
        for row in range(values.shape[0]):
            for column in range(values.shape[1]):
                value = values[row, column]
                label = "n/a" if np.isnan(value) else f"{value:.3f}"
                red, green, blue = to_rgb(colormap(norm(filled[row, column])))
                luminance = 0.299 * red + 0.587 * green + 0.114 * blue
                text_color = "#1c2433" if luminance > 0.55 else "#ffffff"
                axis.text(column, row, label, ha="center", va="center", fontsize=10, fontweight="bold", color=text_color)
        for spine in axis.spines.values():
            spine.set_visible(False)
        st.pyplot(figure, clear_figure=True)

    with ranking_column, st.container(key="card_ranking"):
        section("Ranking by metric")
        focus_metric = st.selectbox("Metric", METRIC_NAMES, index=METRIC_NAMES.index("MCC"))
        ranked = comparison[focus_metric].sort_values()
        figure, axis = styled_figure(palette, size=(4.6, 3.6))
        colors = [palette["accent"] if name == model_name else palette["muted"] for name in ranked.index]
        bars = axis.barh([name.replace("Support Vector Machine", "SVM") for name in ranked.index], ranked.fillna(0).values, color=colors, height=0.6)
        for bar, value in zip(bars, ranked.values):
            label = "n/a" if np.isnan(value) else f"{value:.3f}"
            axis.text(bar.get_width() + 0.005, bar.get_y() + bar.get_height() / 2, label,
                      va="center", fontsize=9, color=palette["text"])
        lower = np.nanmin(ranked.values) if ranked.notna().any() else 0.0
        axis.set_xlim(max(0.0, lower - 0.1), 1.05)
        axis.set_xlabel(focus_metric)
        st.pyplot(figure, clear_figure=True)
        st.caption("The selected model is highlighted.")

with dataset_tab:
    counts = target.value_counts().reindex(CLASS_LABELS, fill_value=0)
    overview = st.columns(4)
    overview[0].metric("Rows", len(data))
    overview[1].metric("Features", len(expected_features))
    overview[2].metric("Benign", int(counts["benign"]))
    overview[3].metric("Malignant", int(counts["malignant"]))

    donut_column, feature_column = st.columns([1, 1.4])
    with donut_column, st.container(key="card_balance"):
        section("Class balance")
        figure, axis = styled_figure(palette, size=(5, 4.2))
        axis.pie(
            counts.values,
            labels=[f"{label}\n{count} ({count / max(1, counts.sum()):.0%})" for label, count in counts.items()],
            colors=[palette["benign"], palette["malignant"]],
            startangle=90,
            wedgeprops={"width": 0.38, "edgecolor": palette["surface"], "linewidth": 2},
            textprops={"color": palette["text"]},
        )
        axis.axis("equal")
        st.pyplot(figure, clear_figure=True)

    with feature_column, st.container(key="card_feature"):
        section("Feature explorer")
        feature = st.selectbox("Feature", expected_features, index=expected_features.index("worst_radius") if "worst_radius" in expected_features else 0)
        figure, axis = styled_figure(palette, size=(7, 4.2))
        for label in CLASS_LABELS:
            values = features.loc[target == label, feature].dropna()
            if len(values):
                axis.hist(values, bins=20, alpha=0.6, color=palette[label], label=label, edgecolor=palette["surface"])
        axis.set_xlabel(feature.replace("_", " "))
        axis.set_ylabel("Count")
        legend = axis.legend(frameon=False)
        for text in legend.get_texts():
            text.set_color(palette["text"])
        st.pyplot(figure, clear_figure=True)

    with st.expander("Feature summary statistics"):
        st.dataframe(features.describe().T.round(3), width="stretch")

    with st.expander("About this project"):
        st.markdown(
            "This dashboard uses the UCI Breast Cancer Wisconsin Diagnostic dataset (569 instances, 30 features). "
            "Each saved pipeline median-imputes missing values, standardizes the features, and applies one of five "
            "classifiers trained on a stratified 80/20 split. Uploaded data is only scored, never used for training. "
            "These models are for coursework and are not intended for clinical diagnosis."
        )

st.markdown(
    '<div class="footer">Machine Learning Assignment 2 · Breast Cancer Classification and Model Comparison Dashboard</div>',
    unsafe_allow_html=True,
)