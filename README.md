# ML Intern Portfolio

Customer **churn classification** project: generate a realistic tabular dataset, compare a logistic regression baseline with a random forest, and save the best model plus metrics.

This is intern-style work: clear problem, sklearn pipelines, train/test split, and metrics you can talk through in an interview.

## Project layout

```
src/ml_intern/     # data, features, train, evaluate
notebooks/         # EDA
tests/             # pipeline smoke tests
data/raw/          # generated churn.csv
models/            # saved joblib model
reports/           # metrics.json
```

## Setup

Python 3.10+ is required. Git is optional until you install it.

```powershell
cd C:\Users\mannu\ml-intern-portfolio
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -e ".[dev]"
```

If `python` opens the Microsoft Store, install Python from [python.org](https://www.python.org/downloads/) and uncheck **App execution aliases** for `python.exe` in Windows Settings.

## Run

```powershell
python -m ml_intern.train
pytest
jupyter notebook notebooks/01_eda.ipynb
```

`python -m ml_intern.train` writes `data/raw/churn.csv`, `models/churn_model.joblib`, and `reports/metrics.json`.

## What the model does

- **Target:** `churned` (1 = customer left)
- **Features:** tenure, monthly charges, support tickets, contract type, addon usage
- **Baseline:** logistic regression with class weights
- **Candidate:** random forest
- **Selection:** higher ROC-AUC on a stratified holdout set

Swap in a real CSV later by dropping it at `data/raw/churn.csv` with the same column names, or change `FEATURE_COLUMNS` in `src/ml_intern/data.py`.
