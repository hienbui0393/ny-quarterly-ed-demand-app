"""Smoke-test every Flask route without touching the real model files.

Run from the project root:
    python tests/smoke_test.py
"""

from __future__ import annotations

import importlib
import os
import sys
import tempfile
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

BASE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE))

FEATURES = [
    "target_lag1",
    "target_lag2",
    "target_lag3",
    "target_lag4",
    "acs_lag2_population_100k",
    "facility_count_lag1",
    "is_nyc",
    "lat",
    "acs_lag2_pct_65plus",
    "acs_lag2_pct_under5",
    "weather_lag4_wet_days",
    "acs_lag2_median_income",
    "lon",
    "is_downstate_non_nyc",
    "acs_lag2_poverty_rate",
    "weather_lag4_snowfall_total",
    "weather_lag4_hot_days",
    "weather_lag4_tavg_mean",
    "weather_lag4_freeze_days",
    "weather_lag4_precip_total",
]
TARGET = "total_ed_encounters"
COUNTIES = [
    ("36001", "Albany County", 42.65, -73.75, 0, 0),
    ("36005", "Bronx County", 40.84, -73.86, 1, 0),
    ("36061", "New York County", 40.78, -73.97, 1, 0),
]


def build_panel() -> pd.DataFrame:
    rng = np.random.default_rng(7)
    rows: list[dict] = []

    for fips, name, lat, lon, is_nyc, downstate in COUNTIES:
        values: list[float] = []
        facility_counts: list[int] = []
        base = 18000 + 25000 * is_nyc + rng.integers(0, 5000)

        for year in range(2021, 2025):
            for quarter_num in range(1, 5):
                quarter = f"Q{quarter_num}"
                seasonal = [0, 900, 300, 1200][quarter_num - 1]
                value = float(base + seasonal + rng.normal(0, 700))
                facility_count = int(3 + is_nyc * 5 + rng.integers(0, 2))

                row = {
                    "fips": fips,
                    "county_name": name,
                    "year": year,
                    "quarter": quarter,
                    TARGET: value,
                    "target_lag1": values[-1] if len(values) >= 1 else np.nan,
                    "target_lag2": values[-2] if len(values) >= 2 else np.nan,
                    "target_lag3": values[-3] if len(values) >= 3 else np.nan,
                    "target_lag4": values[-4] if len(values) >= 4 else np.nan,
                    "facility_count_lag1": facility_counts[-1] if facility_counts else np.nan,
                    "acs_lag2_population_100k": 3.1 + is_nyc * 12,
                    "is_nyc": is_nyc,
                    "lat": lat,
                    "acs_lag2_pct_65plus": 14.0 + rng.random(),
                    "acs_lag2_pct_under5": 5.0 + rng.random(),
                    "weather_lag4_wet_days": 25 + rng.integers(0, 8),
                    "acs_lag2_median_income": 65000 + is_nyc * 18000,
                    "lon": lon,
                    "is_downstate_non_nyc": downstate,
                    "acs_lag2_poverty_rate": 11 + is_nyc * 8,
                    "weather_lag4_snowfall_total": rng.random() * 15,
                    "weather_lag4_hot_days": rng.integers(0, 12),
                    "weather_lag4_tavg_mean": 48 + rng.random() * 8,
                    "weather_lag4_freeze_days": rng.integers(0, 18),
                    "weather_lag4_precip_total": 8 + rng.random() * 5,
                }
                rows.append(row)
                values.append(value)
                facility_counts.append(facility_count)

    return pd.DataFrame(rows)


def build_artifact(panel: pd.DataFrame) -> dict:
    train = panel.dropna(subset=FEATURES + [TARGET])
    model = RandomForestRegressor(n_estimators=30, random_state=42)
    model.fit(train[FEATURES], train[TARGET])
    return {
        "project": "Synthetic quarterly ED smoke test",
        "target": TARGET,
        "target_mode": "level",
        "features": FEATURES,
        "scaler": None,
        "model": model,
        "model_name": "Synthetic Random Forest — level",
        "recommended_method": "Previous-quarter persistence",
        "holdout_year": 2024,
        "data_as_of_date": "2026-07-27",
        "final_metrics": [
            {
                "model": "Previous-quarter persistence",
                "MAE": 1252.6,
                "WAPE": 0.0336,
                "skill_vs_strongest_persistence": 0.0,
            },
            {
                "model": "Tuned XGBoost — level",
                "MAE": 2157.9,
                "WAPE": 0.0578,
                "skill_vs_strongest_persistence": -0.7227,
            },
        ],
    }


def main() -> None:
    with tempfile.TemporaryDirectory() as temp_dir:
        temp = Path(temp_dir)
        panel_path = temp / "county_quarter_analysis.csv"
        artifact_path = temp / "quarterly_ed_forecast_artifact.joblib"

        panel = build_panel()
        panel.to_csv(panel_path, index=False)
        joblib.dump(build_artifact(panel), artifact_path)

        os.environ["PANEL_PATH"] = str(panel_path)
        os.environ["MODEL_PATH"] = str(artifact_path)

        sys.modules.pop("app", None)
        flask_app = importlib.import_module("app")
        client = flask_app.app.test_client()

        health = client.get("/health")
        assert health.status_code == 200, health.get_data(as_text=True)
        assert health.get_json()["status"] == "ok"
        print("GET /health -> 200")

        home = client.get("/")
        assert home.status_code == 200
        assert b"Facility county" in home.data
        print("GET / -> 200")

        historical = client.post(
            "/predict",
            data={"fips": "36001", "period": "2024-Q4"},
        )
        assert historical.status_code == 200
        assert b"Historical comparison" in historical.data
        assert b"Observed encounters" in historical.data
        print("POST /predict historical -> 200")

        future = client.get("/api/predict?fips=36001&period=next")
        assert future.status_code == 200, future.get_json()
        payload = future.get_json()
        assert payload["mode"] == "forecast"
        assert payload["period"] == "2025 Q1"
        assert payload["recommended_prediction"] is not None
        assert payload["ml_prediction"] is None
        print("GET /api/predict next -> 200")

        bad = client.get("/api/predict?fips=99999&period=next")
        assert bad.status_code == 400
        print("Invalid county -> 400")

    print("\nSMOKE TEST PASSED")


if __name__ == "__main__":
    main()
