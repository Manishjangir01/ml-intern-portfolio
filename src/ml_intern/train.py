"""Train baseline and candidate models, persist the best one."""

from __future__ import annotations

import json
from pathlib import Path

import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from ml_intern.data import FEATURE_COLUMNS, TARGET_COLUMN, load_or_create
from ml_intern.evaluate import metrics_dict
from ml_intern.features import build_preprocessor


def build_models() -> dict[str, Pipeline]:
    pre = build_preprocessor()
    return {
        "logreg": Pipeline(
            [
                ("prep", pre),
                ("clf", LogisticRegression(max_iter=1000, class_weight="balanced")),
            ]
        ),
        "random_forest": Pipeline(
            [
                ("prep", build_preprocessor()),
                (
                    "clf",
                    RandomForestClassifier(
                        n_estimators=200,
                        max_depth=8,
                        class_weight="balanced",
                        random_state=42,
                    ),
                ),
            ]
        ),
    }


def train(
    data_path: Path | None = None,
    model_dir: Path = Path("models"),
    reports_dir: Path = Path("reports"),
    random_state: int = 42,
) -> dict:
    df = load_or_create(data_path)
    x = df[FEATURE_COLUMNS]
    y = df[TARGET_COLUMN]
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.2, stratify=y, random_state=random_state
    )

    results = {}
    best_name, best_auc, best_model = None, -1.0, None
    for name, model in build_models().items():
        model.fit(x_train, y_train)
        scores = metrics_dict(model, x_test, y_test)
        results[name] = scores
        if scores["roc_auc"] > best_auc:
            best_name, best_auc, best_model = name, scores["roc_auc"], model

    model_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(best_model, model_dir / "churn_model.joblib")
    summary = {"best_model": best_name, "metrics": results}
    (reports_dir / "metrics.json").write_text(json.dumps(summary, indent=2))
    return summary


if __name__ == "__main__":
    print(json.dumps(train(), indent=2))
