import json

import joblib
import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")  # no display needed in CI

from ml_intern.data import FEATURE_COLUMNS, TARGET_COLUMN, generate_churn_data  # noqa: E402
from ml_intern.evaluate import metrics_dict  # noqa: E402
from ml_intern.features import build_preprocessor  # noqa: E402
from ml_intern.report import make_all  # noqa: E402
from ml_intern.train import build_models, split_data, train  # noqa: E402


def test_generate_churn_data_shape():
    df = generate_churn_data(n_samples=50, random_state=0)
    assert len(df) == 50
    assert set(FEATURE_COLUMNS + [TARGET_COLUMN]).issubset(df.columns)
    assert set(df[TARGET_COLUMN].unique()).issubset({0, 1})


def test_generate_churn_data_is_reproducible():
    a = generate_churn_data(n_samples=100, random_state=7)
    b = generate_churn_data(n_samples=100, random_state=7)
    pd.testing.assert_frame_equal(a, b)


def test_generated_values_are_in_valid_ranges():
    df = generate_churn_data(n_samples=500, random_state=3)
    assert df["tenure_months"].between(1, 72).all()
    assert df["monthly_charges"].between(18, 140).all()
    assert (df["support_tickets"] >= 0).all()
    assert set(df["contract_type"]) <= {"month-to-month", "one-year", "two-year"}


def test_logreg_trains():
    df = generate_churn_data(n_samples=80, random_state=1)
    model = build_models()["logreg"]
    model.fit(df[FEATURE_COLUMNS], df[TARGET_COLUMN])
    preds = model.predict(df[FEATURE_COLUMNS].head(5))
    assert len(preds) == 5


def test_split_is_stratified_and_has_no_overlap():
    df = generate_churn_data(n_samples=400, random_state=2)
    x_train, x_test, y_train, y_test = split_data(df)
    assert len(x_test) == 80
    assert set(x_train.index).isdisjoint(x_test.index)
    assert abs(y_train.mean() - y_test.mean()) < 0.02


def test_metrics_have_expected_keys_and_valid_range():
    df = generate_churn_data(n_samples=300, random_state=4)
    x_train, x_test, y_train, y_test = split_data(df)
    model = build_models()["logreg"].fit(x_train, y_train)
    scores = metrics_dict(model, x_test, y_test)
    assert set(scores) == {"accuracy", "precision", "recall", "f1", "roc_auc"}
    assert all(0.0 <= v <= 1.0 for v in scores.values())
    assert scores["roc_auc"] > 0.5  # better than random guessing


def test_preprocessor_ignores_unseen_contract_type():
    df = generate_churn_data(n_samples=100, random_state=5)
    pre = build_preprocessor().fit(df[FEATURE_COLUMNS])
    new = df[FEATURE_COLUMNS].head(3).copy()
    new["contract_type"] = "three-year"  # category never seen in training
    out = pre.transform(new)
    assert out.shape[0] == 3
    assert not np.isnan(np.asarray(out)).any()


def test_train_saves_model_and_metrics(tmp_path):
    summary = train(
        data_path=tmp_path / "churn.csv",
        model_dir=tmp_path / "models",
        reports_dir=tmp_path / "reports",
    )
    assert summary["best_model"] in {"logreg", "random_forest"}
    assert set(summary["cv_roc_auc"]) == {"logreg", "random_forest"}

    saved = json.loads((tmp_path / "reports" / "metrics.json").read_text())
    assert saved["best_model"] == summary["best_model"]

    model = joblib.load(tmp_path / "models" / "churn_model.joblib")
    df = pd.read_csv(tmp_path / "churn.csv")
    proba = model.predict_proba(df[FEATURE_COLUMNS].head(10))[:, 1]
    assert ((proba >= 0) & (proba <= 1)).all()


def test_report_creates_three_figures(tmp_path):
    train(
        data_path=tmp_path / "churn.csv",
        model_dir=tmp_path / "models",
        reports_dir=tmp_path / "reports",
    )
    paths = make_all(
        out_dir=tmp_path / "figs",
        model_path=tmp_path / "models" / "churn_model.joblib",
        data_path=tmp_path / "churn.csv",
    )
    assert sorted(p.name for p in paths) == [
        "confusion_matrix.png",
        "feature_importance.png",
        "roc_curve.png",
    ]
    assert all(p.exists() and p.stat().st_size > 1000 for p in paths)
