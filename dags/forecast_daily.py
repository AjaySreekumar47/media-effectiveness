from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator
import pandas as pd
import pickle

DATA_PATH = "/opt/airflow/data/features.csv"
MODEL_PATH = "/opt/airflow/artifacts/prophet_preperiod.pkl"
FORECAST_OUT = "/opt/airflow/outputs/forecast_latest.csv"

default_args = {"owner": "you", "retries": 1, "retry_delay": timedelta(minutes=5)}

def forecast_next():
    df = pd.read_csv(DATA_PATH, parse_dates=["date"]).set_index("date")
    with open(MODEL_PATH, "rb") as f:
        bundle = pickle.load(f)
    m, regressors = bundle["model"], bundle["regressors"]
    campaign_days = df[df["brand_burst_flag"]==1].index
    start, end = campaign_days.min(), campaign_days.max()
    post = df.loc[start:end].reset_index().rename(columns={"date":"ds","y":"y"})
    future = post[["ds"] + regressors]
    fc = m.predict(future)
    out = post[["ds","y"]].copy()
    out["yhat"] = fc["yhat"].values
    out["yhat_lower"] = fc["yhat_lower"].values
    out["yhat_upper"] = fc["yhat_upper"].values
    out.to_csv(FORECAST_OUT, index=False)

with DAG(
    dag_id="forecast_daily",
    default_args=default_args,
    schedule_interval="0 6 * * *",  # daily 06:00
    start_date=datetime(2025, 1, 1),
    catchup=False,
    description="Daily forecast push using latest saved model"
) as dag:
    predict = PythonOperator(task_id="forecast_latest", python_callable=forecast_next)
