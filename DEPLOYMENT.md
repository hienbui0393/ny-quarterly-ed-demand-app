# GitHub and Render deployment

This guide applies to the **quarterly facility-county encounter app**. Do not
copy the old annual county-year file names or forecasting logic.

## 1. Final local check

1. Put these files in `model/`:
   - `quarterly_ed_forecast_artifact.joblib`
   - `county_quarter_analysis.csv`
2. Confirm that `.venv`, API keys, tokens, and raw data are absent.
3. Activate the virtual environment and run:

```bash
pip install -r requirements.txt
pip check
python tests/smoke_test.py
python app.py
```

4. Test a historical quarter, the next-quarter result, and `/health`.

## 2. Push to GitHub

Create an empty repository, then run from this folder:

```bash
git init -b main
git add .
git commit -m "Add quarterly ED forecasting app"
git remote add origin https://github.com/YOURNAME/ed-demand-app.git
git push -u origin main
```

Use GitHub authentication or Git Credential Manager when prompted. Do not place
a personal access token inside a saved command, notebook cell, or repository
URL.

## 3. Deploy on Render

Create a **Web Service** from the GitHub repository and use:

- Language: Python 3
- Build command: `pip install -r requirements.txt`
- Start command: `gunicorn --workers 1 --threads 4 app:app`
- Health check path: `/health`

The `.python-version` file selects Python 3.12.8. No Census API key is needed by
the app because it reads the final processed panel.

After deployment, open `/health` first. A successful response confirms that the
server, artifact, and processed panel loaded.
