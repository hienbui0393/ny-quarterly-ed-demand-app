# GitHub and Render deployment

This guide applies to the final quarterly facility-county application.

## 1. Confirm the project files

The `model/` folder must contain:

```text
quarterly_ed_forecast_artifact.joblib
quarterly_ed_xgboost_model.json
county_quarter_analysis.csv
```

The repository must not contain `.venv`, `__pycache__`, API keys, passwords, or
raw unnecessary datasets.

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

## 3. Push changes to GitHub

For an existing repository:

```powershell
git status
git add .
git commit -m "Simplify final forecasting app"
git push
```

Before committing, confirm `.venv` and `__pycache__` are not listed.

## 4. Render configuration

Use these settings:

```text
Language: Python 3
Build command: pip install -r requirements.txt
Start command: gunicorn --workers 1 --threads 4 app:app
Health check path: /health
```

The `.python-version` file requests Python 3.12.8.

## 5. Verify the deployment

Open:

```text
https://ny-quarterly-ed-demand-app.onrender.com/health
```

Confirm that `status` is `ok`, then test the main page and `/compare`.
