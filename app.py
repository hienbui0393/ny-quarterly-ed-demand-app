"""Flask app for the New York quarterly ED-demand capstone.

The app loads the final quarterly artifact and the processed facility-county
panel once at startup. It supports two stakeholder-facing uses:

1. Historical comparison: compare the tuned ML estimate with previous-quarter
   persistence, seasonal persistence, and the observed encounter count.
2. Next-quarter prototype forecast: use the notebook's recommended method.
   In the current project that method is previous-quarter persistence. The ML
   estimate is shown only when a complete future predictor row exists.

Run locally:
    python app.py
"""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from flask import Flask, jsonify, render_template, request
from xgboost import XGBRegressor

BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "model"
MODEL_PATH = Path(
    os.environ.get(
        "MODEL_PATH",
        MODEL_DIR / "quarterly_ed_forecast_artifact.joblib",
    )
)
PANEL_PATH = Path(
    os.environ.get(
        "PANEL_PATH",
        MODEL_DIR / "county_quarter_analysis.csv",
    )
)

app = Flask(__name__)


def load_artifact(path: Path) -> dict[str, Any]:
    """Load artifact metadata and restore the forecasting model."""
    if not path.exists():
        raise FileNotFoundError(
            f"Model artifact not found: {path}. See model/README.md."
        )

    artifact = joblib.load(path)

    required = {
        "features",
        "target",
        "target_mode",
        "recommended_method",
    }
    missing = required - set(artifact)
    if missing:
        raise ValueError(
            f"Artifact is missing required keys: {sorted(missing)}"
        )

    # The production artifact stores XGBoost separately as a portable JSON file.
    # Temporary test artifacts may still contain an embedded model.
    if "model" not in artifact:
        model_file = str(
            artifact.get(
                "model_file",
                "quarterly_ed_xgboost_model.json",
            )
        )
        model_path = path.parent / model_file

        if not model_path.exists():
            raise FileNotFoundError(
                f"XGBoost model file not found: {model_path}"
            )

        model = XGBRegressor()
        model.load_model(model_path)
        artifact["model"] = model

    return artifact


def normalize_fips(value: object) -> str:
    """Return a validated five-digit county FIPS code."""
    text = str(value).strip()
    if not re.fullmatch(r"\d{5}", text):
        raise ValueError("Facility county FIPS must contain exactly five digits.")
    return text


def normalize_quarter(value: object) -> str:
    """Return a validated quarter label such as Q1."""
    text = str(value).strip().upper()
    if text in {"1", "2", "3", "4"}:
        text = f"Q{text}"
    if text not in {"Q1", "Q2", "Q3", "Q4"}:
        raise ValueError(f"Invalid quarter: {value}")
    return text


def period_index(year: int, quarter: str) -> int:
    return int(year) * 4 + int(normalize_quarter(quarter)[1])


def period_label(year: int, quarter: str) -> str:
    return f"{int(year)} {normalize_quarter(quarter)}"


def parse_period(value: str) -> tuple[int, str]:
    """Parse values such as 2024-Q3 or 2024 Q3."""
    match = re.fullmatch(r"\s*(\d{4})[- ](Q[1-4])\s*", value.upper())
    if not match:
        raise ValueError("Period must look like 2024-Q3.")
    return int(match.group(1)), match.group(2)


def load_panel(path: Path, target: str, features: list[str]) -> pd.DataFrame:
    """Load the processed facility-county-quarter analysis panel."""
    if not path.exists():
        raise FileNotFoundError(
            f"Analysis panel not found: {path}. See model/README.md."
        )

    frame = pd.read_csv(path, dtype={"fips": str})
    required = {"fips", "year", "quarter", target, *features}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(
            f"Analysis panel is missing required columns: {sorted(missing)}"
        )

    frame["fips"] = frame["fips"].astype(str).str.zfill(5)
    frame["year"] = pd.to_numeric(frame["year"], errors="raise").astype(int)
    frame["quarter"] = frame["quarter"].map(normalize_quarter)
    frame["period_index"] = [
        period_index(year, quarter)
        for year, quarter in zip(frame["year"], frame["quarter"])
    ]
    frame["period_label"] = [
        period_label(year, quarter)
        for year, quarter in zip(frame["year"], frame["quarter"])
    ]

    if frame.duplicated(["fips", "year", "quarter"]).any():
        raise ValueError("Duplicate facility county-quarter rows were found.")

    return frame.sort_values(["fips", "period_index"]).reset_index(drop=True)


ARTIFACT = load_artifact(MODEL_PATH)
FEATURES = list(ARTIFACT["features"])
TARGET = str(ARTIFACT["target"])
PANEL = load_panel(PANEL_PATH, TARGET, FEATURES)

NAME_COL = next(
    (column for column in ("county_name", "facility_county", "name")
     if column in PANEL.columns),
    None,
)

MODEL_READY = PANEL.dropna(subset=FEATURES + [TARGET]).copy()
if MODEL_READY.empty:
    raise ValueError("The analysis panel has no complete model-ready rows.")

county_columns = ["fips"] + ([NAME_COL] if NAME_COL else [])
COUNTIES = (
    MODEL_READY[county_columns]
    .drop_duplicates("fips")
    .rename(columns={NAME_COL: "name"} if NAME_COL else {})
)
if "name" not in COUNTIES.columns:
    COUNTIES["name"] = COUNTIES["fips"]
COUNTIES = COUNTIES.sort_values("name").reset_index(drop=True)

period_table = (
    MODEL_READY[["year", "quarter", "period_index", "period_label"]]
    .drop_duplicates()
    .sort_values("period_index", ascending=False)
)
HISTORICAL_PERIODS = [
    {"value": f"{row.year}-{row.quarter}", "label": row.period_label}
    for row in period_table.itertuples(index=False)
]


def county_history(fips: str) -> pd.DataFrame:
    fips = normalize_fips(fips)
    history = PANEL[PANEL["fips"].eq(fips)].copy()
    if history.empty:
        raise KeyError(f"Facility county {fips} is not available.")
    return history.sort_values("period_index")


def numeric_feature_frame(row: pd.DataFrame) -> pd.DataFrame:
    """Select model features in their saved order and validate values."""
    values = row.loc[:, FEATURES].apply(pd.to_numeric, errors="raise")
    missing = values.columns[values.isna().any()].tolist()
    if missing:
        raise ValueError(f"Missing model feature values: {missing}")
    if not np.isfinite(values.to_numpy(dtype=float)).all():
        raise ValueError("Model features contain non-finite values.")
    return values


def predict_ml(row: pd.DataFrame) -> float:
    """Run the serialized ML path, including change-target reconstruction."""
    X = numeric_feature_frame(row)
    scaler = ARTIFACT.get("scaler")
    X_fit = scaler.transform(X) if scaler is not None else X
    component = float(np.asarray(ARTIFACT["model"].predict(X_fit))[0])

    if ARTIFACT["target_mode"] == "change":
        lag4 = float(pd.to_numeric(row["target_lag4"], errors="raise").iloc[0])
        return lag4 + component
    return component


def prior_value(row: pd.DataFrame, column: str) -> float | None:
    if column not in row.columns:
        return None
    value = pd.to_numeric(row[column], errors="coerce").iloc[0]
    return float(value) if pd.notna(value) else None


def next_period_from_index(index_value: int) -> tuple[int, str, int]:
    next_index = int(index_value) + 1
    year = (next_index - 1) // 4
    quarter = f"Q{((next_index - 1) % 4) + 1}"
    return year, quarter, next_index


def display_name(row: pd.DataFrame, fips: str) -> str:
    if NAME_COL and NAME_COL in row.columns and pd.notna(row[NAME_COL].iloc[0]):
        return str(row[NAME_COL].iloc[0])
    return fips


def method_prediction(
    method: str,
    ml_prediction: float | None,
    previous_quarter: float | None,
    seasonal: float | None,
) -> float | None:
    if method == "Previous-quarter persistence":
        return previous_quarter
    if method == "Seasonal persistence":
        return seasonal
    return ml_prediction


def historical_result(fips: str, year: int, quarter: str) -> dict[str, Any]:
    history = county_history(fips)
    exact = history[
        history["year"].eq(year) & history["quarter"].eq(quarter)
    ]
    if exact.empty:
        raise ValueError(
            f"{period_label(year, quarter)} is not available for this facility county."
        )

    row = exact.iloc[[-1]].copy()
    if row[FEATURES + [TARGET]].isna().any(axis=None):
        raise ValueError(
            f"{period_label(year, quarter)} does not have a complete model-ready row."
        )

    actual = float(pd.to_numeric(row[TARGET], errors="raise").iloc[0])
    ml_prediction = predict_ml(row)
    previous_quarter = prior_value(row, "target_lag1")
    seasonal = prior_value(row, "target_lag4")
    recommended_method = str(ARTIFACT["recommended_method"])
    recommended_prediction = method_prediction(
        recommended_method,
        ml_prediction,
        previous_quarter,
        seasonal,
    )

    predictions = {
        str(ARTIFACT.get("model_name", "Tuned ML model")): ml_prediction,
        "Previous-quarter persistence": previous_quarter,
        "Seasonal persistence": seasonal,
    }
    errors = [
        {"method": method, "absolute_error": abs(actual - prediction)}
        for method, prediction in predictions.items()
        if prediction is not None
    ]
    errors.sort(key=lambda item: item["absolute_error"])

    return {
        "mode": "historical",
        "fips": fips,
        "county": display_name(row, fips),
        "period": period_label(year, quarter),
        "actual": actual,
        "ml_prediction": ml_prediction,
        "ml_method": str(ARTIFACT.get("model_name", "Tuned ML model")),
        "previous_quarter": previous_quarter,
        "seasonal": seasonal,
        "recommended_method": recommended_method,
        "recommended_prediction": recommended_prediction,
        "errors": errors,
        "best_method": errors[0]["method"] if errors else None,
        "note": (
            "This is a retrospective comparison using a quarter already present "
            "in the processed panel."
        ),
    }


def forecast_result(fips: str) -> dict[str, Any]:
    """Forecast the county's next quarter using the recommended method."""
    history = county_history(fips)
    observed = history.dropna(subset=[TARGET]).sort_values("period_index")
    if observed.empty:
        raise ValueError("No observed encounter history is available.")

    latest = observed.iloc[[-1]].copy()
    next_year, next_quarter, next_index = next_period_from_index(
        int(latest["period_index"].iloc[0])
    )

    previous_quarter = float(latest[TARGET].iloc[0])
    seasonal_row = observed[observed["period_index"].eq(next_index - 4)]
    seasonal = (
        float(seasonal_row[TARGET].iloc[-1])
        if not seasonal_row.empty
        else None
    )

    # Only produce an ML forecast when a complete future predictor row exists.
    future_row = history[history["period_index"].eq(next_index)]
    ml_prediction = None
    if not future_row.empty and not future_row[FEATURES].isna().any(axis=None):
        ml_prediction = predict_ml(future_row.iloc[[-1]])

    recommended_method = str(ARTIFACT["recommended_method"])
    recommended_prediction = method_prediction(
        recommended_method,
        ml_prediction,
        previous_quarter,
        seasonal,
    )
    if recommended_prediction is None:
        raise ValueError(
            "The recommended method requires predictor data that are not yet "
            "available for the next quarter. Refresh the pipeline first."
        )

    note = (
        "The next-quarter recommendation uses the latest observed quarter. "
        "The tuned ML estimate is intentionally omitted until the pipeline "
        "contains a complete future predictor row."
        if ml_prediction is None
        else "A complete future predictor row was available, so the ML estimate is shown."
    )

    return {
        "mode": "forecast",
        "fips": fips,
        "county": display_name(latest, fips),
        "period": period_label(next_year, next_quarter),
        "source_period": str(latest["period_label"].iloc[0]),
        "actual": None,
        "ml_prediction": ml_prediction,
        "ml_method": str(ARTIFACT.get("model_name", "Tuned ML model")),
        "previous_quarter": previous_quarter,
        "seasonal": seasonal,
        "recommended_method": recommended_method,
        "recommended_prediction": recommended_prediction,
        "errors": [],
        "best_method": None,
        "note": note,
    }


def run_prediction(fips: str, selected_period: str) -> dict[str, Any]:
    fips = normalize_fips(fips)
    if selected_period == "next":
        return forecast_result(fips)
    year, quarter = parse_period(selected_period)
    return historical_result(fips, year, quarter)


def final_metric_rows() -> list[dict[str, Any]]:
    rows = []
    for record in ARTIFACT.get("final_metrics", []):
        method = record.get("model") or record.get("Method")
        if not method:
            continue
        rows.append(
            {
                "method": method,
                "mae": record.get("MAE"),
                "wape_percent": (
                    float(record["WAPE"]) * 100
                    if record.get("WAPE") is not None
                    else None
                ),
                "skill": record.get("skill_vs_strongest_persistence"),
            }
        )
    return rows


FINAL_METRICS = final_metric_rows()


@app.template_filter("count")
def format_count(value: object) -> str:
    if value is None or pd.isna(value):
        return "n/a"
    return f"{float(value):,.0f}"


@app.template_filter("decimal")
def format_decimal(value: object) -> str:
    if value is None or pd.isna(value):
        return "n/a"
    return f"{float(value):,.2f}"


@app.get("/")
def index():
    return render_template(
        "index.html",
        counties=COUNTIES.to_dict("records"),
        historical_periods=HISTORICAL_PERIODS,
        metrics=FINAL_METRICS,
        artifact=ARTIFACT,
        n_features=len(FEATURES),
    )


@app.post("/predict")
def predict_form():
    fips = str(request.form.get("fips", ""))
    selected_period = str(request.form.get("period", "next"))
    try:
        result = run_prediction(fips, selected_period)
    except (KeyError, TypeError, ValueError) as exc:
        return render_template("result.html", error=str(exc)), 400
    return render_template("result.html", result=result)


@app.route("/api/predict", methods=["GET", "POST"])
def predict_api():
    """JSON endpoint using `fips` and `period` (`next` or `2024-Q4`)."""
    payload = request.get_json(silent=True)
    if not isinstance(payload, dict):
        payload = request.form if request.form else request.args
    fips = str(payload.get("fips", ""))
    selected_period = str(payload.get("period", "next"))
    try:
        return jsonify(run_prediction(fips, selected_period))
    except (KeyError, TypeError, ValueError) as exc:
        return jsonify({"error": str(exc)}), 400


@app.get("/health")
def health():
    return jsonify(
        {
            "status": "ok",
            "project": ARTIFACT.get("project"),
            "target": TARGET,
            "model_name": ARTIFACT.get("model_name"),
            "recommended_method": ARTIFACT.get("recommended_method"),
            "features": len(FEATURES),
            "facility_counties": int(len(COUNTIES)),
            "panel_periods": [
                str(period_table.iloc[-1]["period_label"]),
                str(period_table.iloc[0]["period_label"]),
            ],
            "data_as_of_date": ARTIFACT.get("data_as_of_date"),
        }
    )


if __name__ == "__main__":
    app.run(
        host="0.0.0.0",
        port=int(os.environ.get("PORT", "5000")),
        debug=os.environ.get("FLASK_DEBUG") == "1",
    )
