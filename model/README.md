# Required model files

Copy exactly these two files from the final Colab project into this folder:

1. `quarterly_ed_forecast_artifact.joblib`  
   Source: `Capstone975/model/quarterly_ed_forecast_artifact.joblib`
2. `county_quarter_analysis.csv`  
   Source: `Capstone975/data/processed/county_quarter_analysis.csv`

Do not use the older annual files or the obsolete quarterly artifacts:

- `best_model_hist_gradient_boosting.joblib`
- `analysis_county_year.csv`
- `ed_demand_county_quarter_model.joblib`
- `quarterly_ed_county_quarter_artifact.joblib`

The app reads the feature list, target definition, fitted model, optional scaler,
final metrics, and recommended method directly from the final artifact.
