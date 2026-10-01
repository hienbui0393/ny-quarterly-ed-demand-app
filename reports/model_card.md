# Model Card: New York Quarterly ED Demand

## Target
Total ED encounters aggregated by facility county and quarter.

## Forecasting use
Retrospective one-quarter-ahead forecasting prototype for staffing and capacity analysis.

## Data period
Facility target years: 2018–2024.
SPARCS snapshot date: 2026-09-22.
Final holdout year: 2024.

## Validation
Expanding-window rolling validation by complete quarter.
Final evaluation on the latest complete holdout year.

## Benchmarks
Previous-quarter persistence and same-quarter previous-year persistence.
Strongest pre-holdout benchmark: Previous-quarter persistence.

## Selected machine-learning model
XGBoost — level

## Selected feature count
20

## Recommended operational method
Previous-quarter persistence

## Important limitations
- Facility county is not necessarily patient residence.
- The per-capita measure is descriptive and is not the forecasting target.
- The final holdout covers one calendar year.
- Publicly suppressed facility cells may understate affected facility county-quarter totals by an unknown amount.
- Pooled R² is secondary because county volumes differ greatly in scale.
