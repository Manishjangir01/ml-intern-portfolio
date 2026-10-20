"""Charts for the README: ROC curves, confusion matrix and feature importance.

Run ``python -m ml_intern.train`` first, then ``python -m ml_intern.report``.
Each function returns a matplotlib Figure so it can also be used in a notebook.
"""

from __future__ import annotations

from pathlib import Path

import joblib
import matplotlib.pyplot as plt
from sklearn.inspection import permutation_importance
from sklearn.metrics import ConfusionMatrixDisplay, RocCurveDisplay

from ml_intern.data import load_or_create
from ml_intern.train import MODEL_DIR, REPORTS_DIR, build_models, split_data

FIGURES_DIR = REPORTS_DIR / "figures"
LABELS = {"logreg": "Logistic regression", "random_forest": "Random forest"}


def plot_roc(x_train, y_train, x_test, y_test):
    """ROC curves of both candidate models on the holdout set."""
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    for name, model in build_models().items():
        model.fit(x_train, y_train)
        RocCurveDisplay.from_estimator(model, x_test, y_test, name=LABELS[name], ax=ax)
    ax.plot([0, 1], [0, 1], "k--", alpha=0.4, label="Random guess")
    ax.set_title("ROC curve (holdout set)")
    ax.legend(loc="lower right")
    fig.tight_layout()
    return fig


def plot_confusion(model, x_test, y_test, title: str = "Confusion matrix (best model)"):
    fig, ax = plt.subplots(figsize=(4.5, 4))
    ConfusionMatrixDisplay.from_estimator(
        model, x_test, y_test, display_labels=["stayed", "churned"], cmap="Blues", ax=ax, colorbar=False
    )
    ax.set_title(title)
    fig.tight_layout()
    return fig


def plot_importance(model, x_test, y_test, random_state: int = 42):
    """Permutation importance: how much ROC-AUC drops when one feature is shuffled."""
    result = permutation_importance(
        model, x_test, y_test, scoring="roc_auc", n_repeats=20, random_state=random_state
    )
    order = result.importances_mean.argsort()
    fig, ax = plt.subplots(figsize=(6, 3.8))
    ax.barh(
        [x_test.columns[i] for i in order],
        result.importances_mean[order],
        xerr=result.importances_std[order],
        color="#2563eb",
    )
    ax.set_xlabel("Drop in ROC-AUC when shuffled")
    ax.set_title("Feature importance (best model)")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    return fig


def make_all(
    out_dir: Path = FIGURES_DIR,
    model_path: Path = MODEL_DIR / "churn_model.joblib",
    data_path: Path | None = None,
) -> list[Path]:
    df = load_or_create(data_path)
    x_train, x_test, y_train, y_test = split_data(df)
    best = joblib.load(model_path)

    out_dir.mkdir(parents=True, exist_ok=True)
    figures = {
        "roc_curve.png": plot_roc(x_train, y_train, x_test, y_test),
        "confusion_matrix.png": plot_confusion(best, x_test, y_test),
        "feature_importance.png": plot_importance(best, x_test, y_test),
    }
    paths = []
    for filename, fig in figures.items():
        path = out_dir / filename
        fig.savefig(path, dpi=150)
        plt.close(fig)
        paths.append(path)
    return paths


if __name__ == "__main__":
    for p in make_all():
        print("Saved", p)
