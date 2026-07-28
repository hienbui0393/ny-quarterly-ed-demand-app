# GitHub and Render deployment

This guide applies to the **quarterly facility-county encounter app**. Do not
use the old annual county-year filenames or forecasting logic.

## 1. Preserve the three model files

Before replacing an older app folder, keep these files from the working
`model/` folder:

1. `quarterly_ed_forecast_artifact.joblib`
2. `quarterly_ed_xgboost_model.json`
3. `county_quarter_analysis.csv`

Copy them into the updated app's `model/` folder.

## 2. Final local check

```powershell
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
pip check
python tests\smoke_test.py
python tests\production_load_test.py
python app.py
```

Test:

- a historical quarter;
- the one-quarter-ahead prototype;
- `/compare` with two or more counties;
- `/health`.

## 3. Push updates to GitHub

From the repository root:

```powershell
git add .
git status
git commit -m "Improve forecast comparison and app limitations"
git push
```

Confirm `.venv`, tokens, API keys, and raw data are not listed by `git status`.
Do not place a personal access token in a saved command or repository URL.

## 4. Render settings

Use:

- Language: Python 3
- Build command: `pip install -r requirements.txt`
- Start command: `gunicorn --workers 1 --threads 4 app:app`
- Health check path: `/health`

The `.python-version` file selects Python 3.12.8. No Census API key is required
by the Flask app because it reads the final processed panel and saved model.

After deployment, test:

- `/health`
- `/`
- `/compare`
- `/api/predict?fips=36001&period=2024-Q4&method=ml`
