# New York Quarterly ED Demand App

A simple Flask app for the DATA 975 capstone. It presents total emergency
department encounters by **facility county and quarter** and compares:

- the tuned XGBoost estimate;
- previous-quarter persistence;
- same-quarter previous-year persistence.

The notebook recommends previous-quarter persistence, so the app uses it as the
primary next-quarter prototype forecast and keeps the ML estimate visible for
historical comparison.

## Recommended structure

For the simplest GitHub and Render setup, use this application folder as the
repository root:

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
│   ├── county_quarter_analysis.csv
│   └── README.md
├── templates/
│   ├── base.html
│   ├── index.html
│   └── result.html
├── static/
│   └── style.css
└── tests/
    └── smoke_test.py
```

You may add `notebooks/ed_demand_pipeline.ipynb` to the same repository for your
portfolio. The app does not need the notebook to run.

## 1. Add the two final files

Follow `model/README.md`. Do not copy the older annual artifact or obsolete
quarterly artifacts. The application will not start until both required files
are present.

## 2. Create a local virtual environment

A `.venv` is recommended locally, but it must not be uploaded to GitHub or
Render. It is already excluded by `.gitignore`.

### Windows PowerShell

```powershell
py -3.12 -m venv .venv
.venv\Scripts\Activate.ps1
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

## 3. Test and run locally

```bash
python tests/smoke_test.py
python app.py
```

Open `http://127.0.0.1:5000` and test:

- one historical quarter;
- the next-quarter result;
- `http://127.0.0.1:5000/health`.

The smoke test uses temporary synthetic files and does not overwrite the real
artifact or panel.

## App behavior

- **Historical quarter:** shows observed encounters, tuned ML,
  previous-quarter persistence, seasonal persistence, and absolute errors.
- **Next quarter:** uses the artifact's recommended method. The app does not
  invent an ML estimate when a complete future predictor row is unavailable.

This is a retrospective public-data prototype. A live staffing system would
require a current internal encounter feed and a refreshed predictor pipeline.

See `DEPLOYMENT.md` for the short GitHub and Render workflow.
