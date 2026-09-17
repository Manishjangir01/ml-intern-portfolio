from ml_intern.data import generate_churn_data, FEATURE_COLUMNS, TARGET_COLUMN
from ml_intern.train import build_models


def test_generate_churn_data_shape():
    df = generate_churn_data(n_samples=50, random_state=0)
    assert len(df) == 50
    assert set(FEATURE_COLUMNS + [TARGET_COLUMN]).issubset(df.columns)
    assert set(df[TARGET_COLUMN].unique()).issubset({0, 1})


def test_logreg_trains():
    df = generate_churn_data(n_samples=80, random_state=1)
    model = build_models()["logreg"]
    model.fit(df[FEATURE_COLUMNS], df[TARGET_COLUMN])
    preds = model.predict(df[FEATURE_COLUMNS].head(5))
    assert len(preds) == 5
