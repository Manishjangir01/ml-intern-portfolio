"""Synthetic customer churn dataset (no external download required)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

# Project root (two levels above src/ml_intern), so paths work from any directory.
ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA_PATH = ROOT / "data" / "raw" / "churn.csv"

FEATURE_COLUMNS = [
    "tenure_months",
    "monthly_charges",
    "support_tickets",
    "contract_type",
    "uses_addon",
]
TARGET_COLUMN = "churned"
CATEGORICAL_COLUMNS = ["contract_type"]
NUMERIC_COLUMNS = ["tenure_months", "monthly_charges", "support_tickets", "uses_addon"]


def generate_churn_data(n_samples: int = 1200, random_state: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(random_state)
    tenure = rng.integers(1, 73, size=n_samples)
    monthly = rng.normal(65, 22, size=n_samples).clip(18, 140)
    tickets = rng.poisson(1.4, size=n_samples).clip(0, 12)
    contract = rng.choice(["month-to-month", "one-year", "two-year"], size=n_samples, p=[0.55, 0.28, 0.17])
    addon = rng.integers(0, 2, size=n_samples)

    logit = (
        -1.8
        + 0.035 * monthly
        + 0.28 * tickets
        - 0.045 * tenure
        - 0.7 * addon
        + np.where(contract == "month-to-month", 0.9, 0.0)
        + np.where(contract == "two-year", -0.8, 0.0)
    )
    prob = 1 / (1 + np.exp(-logit))
    churned = rng.binomial(1, prob)

    return pd.DataFrame(
        {
            "tenure_months": tenure,
            "monthly_charges": monthly.round(2),
            "support_tickets": tickets,
            "contract_type": contract,
            "uses_addon": addon,
            "churned": churned,
        }
    )


def load_or_create(path: Path | None = None, **kwargs) -> pd.DataFrame:
    if path is None:
        path = DEFAULT_DATA_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        return pd.read_csv(path)
    df = generate_churn_data(**kwargs)
    df.to_csv(path, index=False)
    return df
