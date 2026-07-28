"""Verify that the real production files load correctly.

Run from the project root after the three files are in model/:
    python tests/production_load_test.py
"""

from __future__ import annotations

import sys
from pathlib import Path

BASE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE))

import app  # noqa: E402

assert app.ARTIFACT["model"].__class__.__name__ == "XGBRegressor"
assert len(app.FEATURES) > 0
assert not app.PANEL.empty
assert len(app.COUNTIES) > 0

print("Production model:", app.ARTIFACT.get("model_name"))
print("Features:", len(app.FEATURES))
print("Facility counties:", len(app.COUNTIES))
print("Panel rows:", len(app.PANEL))
print("PRODUCTION MODEL LOAD TEST PASSED")
