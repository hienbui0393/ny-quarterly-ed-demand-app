# New York Quarterly ED Demand App

A focused Flask application for the DATA 975 capstone. It presents total
emergency department encounters by **facility county and quarter** and compares:

- the tuned XGBoost estimate;
- previous-quarter persistence;
- same-quarter previous-year persistence.

The final analysis recommends previous-quarter persistence, so the application
uses it for the one-quarter-ahead prototype and retains XGBoost for historical
evaluation.

## Live application

https://ny-quarterly-ed-demand-app.onrender.com/

## Final design

The application intentionally remains simple:

- **Main page:** choose one county and one period.
- **Prototype forecast:** uses the notebook's recommended method.
- **Historical result:** shows observed demand, XGBoost, and both persistence
  benchmarks.
- **Multi-county evaluation:** compares XGBoost with previous-quarter
  persistence across two to ten counties.
- **Limitations banner:** prevents the retrospective result from being read as a
  live current-quarter forecast.

The application does **not** include a Ridge/Random Forest/XGBoost selector or a
map. XGBoost is the single deployed ML model selected by the final notebook.

## Project scope

- Unit: one New York facility county-quarter
- Target: total ED encounters
- Deployed model-ready panel: 2019 Q1 through 2024 Q4
- Selected ML model: XGBoost — level
- Recommended method: previous-quarter persistence
- Purpose: retrospective planning prototype, not a live staffing feed

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
├── notebooks/
│   └── ed_demand_pipeline.ipynb
├── templates/
│   ├── base.html
│   ├── index.html
│   ├── result.html
│   └── compare.html
├── static/
│   └── style.css
└── tests/
    └── smoke_test.py
```

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
python app.py
```

Open:

- `http://127.0.0.1:5000/`
- `http://127.0.0.1:5000/compare`
- `http://127.0.0.1:5000/health`

## Interpretation

Skill is the percentage reduction in MAE relative to previous-quarter
persistence. Positive skill favors XGBoost; negative skill favors persistence.

## Limitations

This is a retrospective public-data prototype. Facility county is not
necessarily the patient's county of residence. Public suppression may
understate totals in affected facility county-quarters. A live operational
version would require a current internal encounter feed and refreshed
predictors.
