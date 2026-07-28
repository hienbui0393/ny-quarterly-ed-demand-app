"""Flask app for the New York quarterly ED-demand capstone.

The app loads the final quarterly artifact and processed facility-county panel
once at startup. It supports:

1. A one-quarter-ahead prototype forecast using the notebook's recommended
   method.
2. Historical comparison of tuned XGBoost, previous-quarter persistence, and
   seasonal persistence.
3. A compact multi-county comparison that reports pooled skill against
   previous-quarter persistence.

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

    # The production artifact stores XGBoost separately in portable JSON.
    # Synthetic test artifacts may still contain an embedded estimator.
    if "model" not in artifact:
        model_file = str(
            artifact.get("model_file", "quarterly_ed_xgboost_model.json")
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
    """Parse a value such as 2024-Q3 or 2024 Q3."""
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
    (
        column
        for column in ("county_name", "facility_county", "name")
        if column in PANEL.columns
    ),
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

ML_METHOD = str(ARTIFACT.get("model_name", "Tuned XGBoost"))
PREVIOUS_METHOD = "Previous-quarter persistence"
SEASONAL_METHOD = "Seasonal persistence"
METHOD_OPTIONS = [
    {"value": "recommended", "label": f"Recommended — {ARTIFACT['recommended_method']}"},
    {"value": "ml", "label": ML_METHOD},
    {"value": "previous", "label": PREVIOUS_METHOD},
    {"value": "seasonal", "label": SEASONAL_METHOD},
]
COMPARE_METHOD_OPTIONS = [
    {"value": "ml", "label": ML_METHOD},
    {"value": "seasonal", "label": SEASONAL_METHOD},
]


def county_history(fips: str) -> pd.DataFrame:
    fips = normalize_fips(fips)
    history = PANEL[PANEL["fips"].eq(fips)].copy()
    if history.empty:
        raise KeyError(f"Facility county {fips} is not available.")
    return history.sort_values("period_index")


def numeric_feature_frame(row: pd.DataFrame) -> pd.DataFrame:
    """Select model features in saved order and validate numeric values."""
    values = row.loc[:, FEATURES].apply(pd.to_numeric, errors="raise")
    missing = values.columns[values.isna().any()].tolist()
    if missing:
        raise ValueError(f"Missing model feature values: {missing}")
    if not np.isfinite(values.to_numpy(dtype=float)).all():
        raise ValueError("Model features contain non-finite values.")
    return values


def predict_ml(row: pd.DataFrame) -> float:
    """Run the saved ML path, including change-target reconstruction."""
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


def resolve_method_label(method_key: str) -> str:
    mapping = {
        "recommended": str(ARTIFACT["recommended_method"]),
        "ml": ML_METHOD,
        "previous": PREVIOUS_METHOD,
        "seasonal": SEASONAL_METHOD,
    }
    if method_key not in mapping:
        raise ValueError("Unknown forecasting method selection.")
    return mapping[method_key]


def prediction_for_method(
    method_label: str,
    ml_prediction: float | None,
    previous_quarter: float | None,
    seasonal: float | None,
) -> float | None:
    if method_label == PREVIOUS_METHOD:
        return previous_quarter
    if method_label == SEASONAL_METHOD:
        return seasonal
    return ml_prediction


def method_cards(
    *,
    ml_prediction: float | None,
    previous_quarter: float | None,
    seasonal: float | None,
    selected_method: str,
) -> list[dict[str, Any]]:
    values = [
        (ML_METHOD, ml_prediction, "Tuned ML estimate"),
        (PREVIOUS_METHOD, previous_quarter, "Previous observed quarter"),
        (SEASONAL_METHOD, seasonal, "Same quarter in the previous year"),
    ]
    return [
        {
            "label": label,
            "prediction": prediction,
            "detail": detail,
            "selected": label == selected_method,
            "recommended": label == str(ARTIFACT["recommended_method"]),
        }
        for label, prediction, detail in values
        if label != selected_method
    ]


def historical_result(
    fips: str,
    year: int,
    quarter: str,
    method_key: str = "recommended",
) -> dict[str, Any]:
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
    selected_method = resolve_method_label(method_key)
    selected_prediction = prediction_for_method(
        selected_method,
        ml_prediction,
        previous_quarter,
        seasonal,
    )
    if selected_prediction is None:
        raise ValueError(f"{selected_method} is unavailable for this period.")

    predictions = {
        ML_METHOD: ml_prediction,
        PREVIOUS_METHOD: previous_quarter,
        SEASONAL_METHOD: seasonal,
    }
    errors = [
        {
            "method": method,
            "absolute_error": abs(actual - prediction),
            "recommended": method == str(ARTIFACT["recommended_method"]),
            "selected": method == selected_method,
        }
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
        "selected_method": selected_method,
        "selected_prediction": selected_prediction,
        "selected_error": abs(actual - selected_prediction),
        "recommended_method": str(ARTIFACT["recommended_method"]),
        "cards": method_cards(
            ml_prediction=ml_prediction,
            previous_quarter=previous_quarter,
            seasonal=seasonal,
            selected_method=selected_method,
        ),
        "errors": errors,
        "best_method": errors[0]["method"] if errors else None,
        "note": (
            "This is a retrospective comparison using a quarter already present "
            "in the processed panel."
        ),
    }


def forecast_result(
    fips: str,
    method_key: str = "recommended",
) -> dict[str, Any]:
    """Forecast the first quarter after the latest observed panel period."""
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

    # ML is produced only when the panel already contains a complete future row.
    future_row = history[history["period_index"].eq(next_index)]
    ml_prediction = None
    if not future_row.empty and not future_row[FEATURES].isna().any(axis=None):
        ml_prediction = predict_ml(future_row.iloc[[-1]])

    selected_method = resolve_method_label(method_key)
    selected_prediction = prediction_for_method(
        selected_method,
        ml_prediction,
        previous_quarter,
        seasonal,
    )
    if selected_prediction is None:
        if selected_method == ML_METHOD:
            raise ValueError(
                "The tuned XGBoost estimate is unavailable for the next panel "
                "quarter because future ACS and weather predictors have not been "
                "refreshed. Choose the recommended or a persistence method."
            )
        raise ValueError(f"{selected_method} is unavailable for the next quarter.")

    note = (
        "This is a one-quarter-ahead prototype forecast after the latest available "
        "public-data period, not a live forecast for the current calendar quarter."
    )

    return {
        "mode": "forecast",
        "fips": fips,
        "county": display_name(latest, fips),
        "period": period_label(next_year, next_quarter),
        "source_period": str(latest["period_label"].iloc[0]),
        "actual": None,
        "selected_method": selected_method,
        "selected_prediction": selected_prediction,
        "recommended_method": str(ARTIFACT["recommended_method"]),
        "cards": method_cards(
            ml_prediction=ml_prediction,
            previous_quarter=previous_quarter,
            seasonal=seasonal,
            selected_method=selected_method,
        ),
        "errors": [],
        "best_method": None,
        "note": note,
    }


def run_prediction(
    fips: str,
    selected_period: str,
    method_key: str = "recommended",
) -> dict[str, Any]:
    fips = normalize_fips(fips)
    if selected_period == "next":
        return forecast_result(fips, method_key)
    year, quarter = parse_period(selected_period)
    return historical_result(fips, year, quarter, method_key)


def compare_historical(
    fips_values: list[str],
    selected_period: str,
    method_key: str,
) -> dict[str, Any]:
    """Compare one method with previous-quarter persistence across counties."""
    unique_fips = list(dict.fromkeys(normalize_fips(value) for value in fips_values))
    if len(unique_fips) < 2:
        raise ValueError("Select at least two facility counties.")
    if len(unique_fips) > 10:
        raise ValueError("Select no more than ten facility counties at a time.")

    if selected_period == "next":
        raise ValueError("The multi-county comparison requires a historical quarter.")

    year, quarter = parse_period(selected_period)
    selected_method = resolve_method_label(method_key)
    if selected_method == PREVIOUS_METHOD:
        raise ValueError(
            "Choose XGBoost or seasonal persistence; previous-quarter persistence "
            "is already the comparison benchmark."
        )

    rows: list[dict[str, Any]] = []
    for fips in unique_fips:
        result = historical_result(fips, year, quarter, method_key)
        previous_card = next(
            (
                item
                for item in result["cards"]
                if item["label"] == PREVIOUS_METHOD
            ),
            None,
        )
        previous_prediction = (
            result["selected_prediction"]
            if result["selected_method"] == PREVIOUS_METHOD
            else previous_card["prediction"] if previous_card else None
        )
        if previous_prediction is None:
            continue

        actual = float(result["actual"])
        selected_prediction = float(result["selected_prediction"])
        selected_error = abs(actual - selected_prediction)
        previous_error = abs(actual - float(previous_prediction))
        row_skill = (
            1 - selected_error / previous_error
            if previous_error > 0
            else None
        )
        rows.append(
            {
                "county": result["county"],
                "fips": fips,
                "actual": actual,
                "selected_prediction": selected_prediction,
                "previous_prediction": float(previous_prediction),
                "selected_error": selected_error,
                "previous_error": previous_error,
                "skill_percent": row_skill * 100 if row_skill is not None else None,
                "selected_wins": selected_error < previous_error,
            }
        )

    if not rows:
        raise ValueError("No complete county rows were available for comparison.")

    selected_mae = float(np.mean([row["selected_error"] for row in rows]))
    previous_mae = float(np.mean([row["previous_error"] for row in rows]))
    pooled_skill = (
        1 - selected_mae / previous_mae
        if previous_mae > 0
        else None
    )

    return {
        "period": period_label(year, quarter),
        "selected_method": selected_method,
        "benchmark_method": PREVIOUS_METHOD,
        "rows": rows,
        "county_count": len(rows),
        "selected_mae": selected_mae,
        "previous_mae": previous_mae,
        "skill_percent": pooled_skill * 100 if pooled_skill is not None else None,
        "wins": sum(row["selected_wins"] for row in rows),
    }


def final_metric_rows() -> list[dict[str, Any]]:
    rows = []
    for record in ARTIFACT.get("final_metrics", []):
        method = record.get("model") or record.get("Method")
        if not method:
            continue
        skill = record.get("skill_vs_strongest_persistence")
        rows.append(
            {
                "method": method,
                "mae": record.get("MAE"),
                "wape_percent": (
                    float(record["WAPE"]) * 100
                    if record.get("WAPE") is not None
                    else None
                ),
                "skill_percent": float(skill) * 100 if skill is not None else None,
            }
        )
    return rows


FINAL_METRICS = final_metric_rows()


@app.template_filter("count")
def format_count(value: object) -> str:
    if value is None or pd.isna(value):
        return "Unavailable"
    return f"{float(value):,.0f}"


@app.template_filter("decimal")
def format_decimal(value: object) -> str:
    if value is None or pd.isna(value):
        return "Unavailable"
    return f"{float(value):,.2f}"


@app.get("/")
def index():
    return render_template(
        "index.html",
        counties=COUNTIES.to_dict("records"),
        historical_periods=HISTORICAL_PERIODS,
        method_options=METHOD_OPTIONS,
        metrics=FINAL_METRICS,
        artifact=ARTIFACT,
        n_features=len(FEATURES),
        panel_start=str(period_table.iloc[-1]["period_label"]),
        panel_end=str(period_table.iloc[0]["period_label"]),
    )


@app.post("/predict")
def predict_form():
    fips = str(request.form.get("fips", ""))
    selected_period = str(request.form.get("period", "next"))
    method_key = str(request.form.get("method", "recommended"))
    try:
        result = run_prediction(fips, selected_period, method_key)
    except (KeyError, TypeError, ValueError) as exc:
        return render_template("result.html", error=str(exc)), 400
    return render_template("result.html", result=result)


@app.route("/compare", methods=["GET", "POST"])
def compare():
    comparison = None
    error = None
    selected_period = HISTORICAL_PERIODS[0]["value"]
    selected_method = "ml"
    selected_counties: list[str] = []

    if request.method == "POST":
        selected_period = str(request.form.get("period", selected_period))
        selected_method = str(request.form.get("method", "ml"))
        selected_counties = request.form.getlist("fips")
        try:
            comparison = compare_historical(
                selected_counties,
                selected_period,
                selected_method,
            )
        except (KeyError, TypeError, ValueError) as exc:
            error = str(exc)

    return render_template(
        "compare.html",
        counties=COUNTIES.to_dict("records"),
        historical_periods=HISTORICAL_PERIODS,
        method_options=COMPARE_METHOD_OPTIONS,
        comparison=comparison,
        error=error,
        selected_period=selected_period,
        selected_method=selected_method,
        selected_counties=selected_counties,
    ), 400 if error else 200


@app.route("/api/predict", methods=["GET", "POST"])
def predict_api():
    """JSON endpoint using fips, period, and optional method."""
    payload = request.get_json(silent=True)
    if not isinstance(payload, dict):
        payload = request.form if request.form else request.args
    fips = str(payload.get("fips", ""))
    selected_period = str(payload.get("period", "next"))
    method_key = str(payload.get("method", "recommended"))
    try:
        return jsonify(run_prediction(fips, selected_period, method_key))
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
