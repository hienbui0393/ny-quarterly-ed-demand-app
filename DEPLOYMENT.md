# GitHub and Render Deployment

This guide preserves the original Flask deployment and updates only the processed
data location for the SQL-refactored project.

## 1. Required generated files before deployment

After running SQL + the modeling notebook, confirm:

```text
model/
├── quarterly_ed_forecast_artifact.joblib
└── quarterly_ed_xgboost_model.json

data/processed/
└── county_quarter_analysis.csv
```

The repository must not contain `.venv`, `__pycache__`, API keys, passwords, or
unnecessary raw datasets.

## 2. Test locally

From the project root:

```powershell
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
pip check
python tests\smoke_test.py
python app.py
```

Test:

- one historical quarter;
- the one-quarter-ahead prototype;
- `/compare` with two counties;
- `/health`.

Stop the server with `Ctrl + C`.

## 3. Confirm app paths

`app.py` loads:

- model metadata from `model/quarterly_ed_forecast_artifact.joblib`;
- XGBoost from `model/quarterly_ed_xgboost_model.json`;
- processed panel from `data/processed/county_quarter_analysis.csv`.

Environment variables `MODEL_PATH` and `PANEL_PATH` can override the defaults.

## 4. Push changes to GitHub

```powershell
git status
git add .
git commit -m "Refactor ED pipeline with Oracle SQL and BI layer"
git push
```

Before committing, confirm `.venv`, `__pycache__`, and raw source data are not
listed.

## 5. Render configuration

Use the existing settings:

```text
Language: Python 3
Build command: pip install -r requirements.txt
Start command: gunicorn --workers 1 --threads 4 app:app
Health check path: /health
```

`.python-version` requests Python 3.12.8.

## 6. Verify deployment

Open:

```text
https://ny-quarterly-ed-demand-app.onrender.com/health
```

Confirm that `status` is `ok`, then test the main page and `/compare`.

## Important deployment note

The deployed Flask service does not need a live Oracle connection. Oracle is used
upstream to prepare the versioned analytical panel, which is exported as the CSV
that the application loads. This keeps deployment simple while the portfolio still
demonstrates Oracle SQL, Python ML, Power BI, and Flask.
