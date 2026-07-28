# New York Quarterly ED Demand App

A simple Flask application for the DATA 975 capstone. It presents total
emergency department encounters by **facility county and quarter** and compares:

- the tuned XGBoost estimate;
- previous-quarter persistence;
- same-quarter previous-year persistence.

The notebook recommends previous-quarter persistence, so the app uses it as the
default prototype forecast and keeps the ML estimate visible for historical
comparison.

## Live application

https://ny-quarterly-ed-demand-app.onrender.com/

## Project scope

- Unit: one New York facility county-quarter
- Target: total ED encounters
- Panel: 2019 Q1 through 2024 Q4 in the deployed snapshot
- Selected ML model: XGBoost — level
- Recommended method: previous-quarter persistence
- Deployment purpose: retrospective planning prototype, not a live staffing feed

## Main features

- **One-quarter-ahead prototype:** forecasts the first quarter after the latest
  available panel period using the selected method.
- **Historical comparison:** shows observed encounters, XGBoost, previous-quarter
  persistence, seasonal persistence, and absolute errors.
- **Method selector:** lets a reviewer emphasize the recommended method, XGBoost,
  previous-quarter persistence, or seasonal persistence.
- **Multi-county comparison:** compares XGBoost or seasonal persistence with
  previous-quarter persistence across two to ten counties and reports pooled
  skill.
- **Health and JSON endpoints:** `/health` and `/api/predict`.
- **Clear limitations:** explains facility-county scope, public suppression, and
  the difference between the latest panel quarter and the current calendar date.

## Repository structure

```text
ed-demand-app/
├── app.py
├── README.md
├── DEPLOYMENT.md
├── requirements.txt
├── .python-version
├── .gitignore
├── Procfile
├── model/
│   ├── quarterly_ed_forecast_artifact.joblib
│   ├── quarterly_ed_xgboost_model.json
│   ├── county_quarter_analysis.csv
│   └── README.md
├── templates/
│   ├── base.html
│   ├── index.html
│   ├── result.html
│   └── compare.html
├── static/
│   └── style.css
└── tests/
    ├── smoke_test.py
    └── production_load_test.py
```

The model files are project-specific outputs from the final Colab notebook. See
`model/README.md`.

## Local setup

### Windows PowerShell

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
pip check
```

### macOS or Linux

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
pip check
```

## Test and run

```bash
python tests/smoke_test.py
python tests/production_load_test.py
python app.py
```

Open `http://127.0.0.1:5000` and test:

- one historical quarter;
- the one-quarter-ahead prototype;
- the multi-county comparison;
- `http://127.0.0.1:5000/health`.

## Interpretation

Positive skill means the selected method improves on previous-quarter
persistence; negative skill means it performs worse. The app reports skill as a
percentage to make the comparison easier to read.

## Limitations

This is a retrospective public-data prototype. Facility county is not
necessarily the patient's county of residence. Publicly suppressed facility
values may understate totals in affected facility county-quarters. A live
operational deployment would require a current internal encounter feed and a
refreshed predictor pipeline.
