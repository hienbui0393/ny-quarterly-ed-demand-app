# Quarterly ED Demand Results

- SPARCS snapshot date: 2026-09-22
- Final holdout year: 2024
- Selected ML candidate: XGBoost — level
- Target formulation: level
- Selected feature count: 20
- Strongest persistence benchmark: Previous-quarter persistence
- Recommended operational method: Previous-quarter persistence
- Selected features: target_lag1, target_lag2, target_lag3, target_lag4, acs_lag2_population_100k, facility_count_lag1, is_nyc, lat, acs_lag2_pct_65plus, acs_lag2_pct_under5, weather_lag4_wet_days, acs_lag2_median_income, lon, is_downstate_non_nyc, acs_lag2_poverty_rate, weather_lag4_snowfall_total, weather_lag4_hot_days, weather_lag4_tavg_mean, weather_lag4_freeze_days, weather_lag4_precip_total

## Final holdout comparison

| Method | MAE | RMSE | WAPE (%) | Level R² | Skill |
|---|---:|---:|---:|---:|---:|
| Previous-quarter persistence | 1,252.6 | 3,421.1 | 3.36 | 0.9970 | 0.0000 |
| Tuned XGBoost — level | 2,144.0 | 5,204.9 | 5.75 | 0.9930 | -0.7116 |
| Seasonal persistence | 2,840.8 | 7,723.1 | 7.61 | 0.9846 | -1.2679 |

Skill is measured against Previous-quarter persistence. Positive skill
means the method reduces MAE relative to that benchmark; negative
skill means it performs worse.

## Week of July 14 — model family carried forward

XGBoost — level is the machine-learning specification carried forward to tuning because it achieved the lowest mean rolling-origin MAE among the ML candidates at 2,991.4 encounters, compared with 2,992.7 for Random Forest — level. This selection identifies the strongest ML candidate under the common validation design; it does not imply that ML beat persistence overall.

## Week of July 21 — when the model adds value over persistence

XGBoost — level had mean rolling MAE 2,991.4, compared with 2,419.0 for Previous-quarter persistence. It beat persistence in 4 of 12 quarters: 2021 Q4 (0.002 skill), 2022 Q2 (0.104 skill), 2023 Q1 (0.107 skill), 2023 Q2 (0.001 skill). Therefore, Previous-quarter persistence remains the stronger overall forecasting method when its mean MAE is lower.

## Final recommendation

The tuned ML model did not beat Previous-quarter persistence on the final holdout. Benchmark MAE was 1,253; ML MAE was 2,144. Use persistence as the primary prototype forecast and retain ML for comparison and diagnostics.

## Important limitations

- Facility county is not necessarily patient county of residence.
- The final holdout contains only one complete calendar year.
- Publicly suppressed facility cells may understate affected totals by an unknown amount.
- The largest absolute errors occur in high-volume urban counties.
- Pooled level R² is secondary to out-of-time MAE and skill.
- Persistence remains the operational recommendation whenever it
  outperforms the machine-learning model.
