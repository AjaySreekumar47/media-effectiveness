# Media Effectiveness: Forecasting, Causal Impact & Attribution

**What this repo contains**
- Phase 0–5 notebooks (simulation → features → SARIMAX/Prophet → Causal Impact → ML attribution → reporting)
- Streamlit app (`app.py`) with tabs, scenario simulator, and uncertainty bands
- Airflow DAGs for weekly retrains and daily forecast pushes

## Quickstart (Streamlit locally)
```bash
pip install -r requirements.txt
streamlit run app.py
````

## Project structure

```
media-effectiveness/
├─ app.py
├─ src/
│  ├─ utils.py
│  └─ scenario.py
├─ notebooks/
│  ├─ 01_features_and_diagnostics.ipynb
│  ├─ 02_model_sarimax.ipynb
│  ├─ 03_causal_inference.ipynb
│  ├─ 04_ml_channel_attribution.ipynb
│  └─ 05_reporting_integration.ipynb
├─ data/
│  ├─ features.csv
│  └─ simulated_media.csv
├─ dags/
│  ├─ retrain_weekly.py
│  └─ forecast_daily.py
├─ artifacts/        # model artifacts if you choose to commit small ones
├─ outputs/          # optional forecasts/plots
├─ requirements.txt
├─ LICENSE
└─ README.md
```

## Reproduce the analysis

1. Open `notebooks/01_...` and run cells in order by phase, or
2. Use the Streamlit app as a demo UI.

## Airflow

* `dags/retrain_weekly.py`: trains Prophet on pre-period weekly (cron `0 3 * * 0`)
* `dags/forecast_daily.py`: pushes daily forecasts with uncertainty (cron `0 6 * * *`)
  Update file paths to your environment (`/opt/airflow/...` by default).

## Notes

* Data are simulated for portfolio purposes.
* iROAS < 1 in the demo shows how positive lift can still be inefficient.
