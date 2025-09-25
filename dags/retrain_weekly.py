from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator
import pandas as pd
import pickle
from prophet import Prophet

DATA_PATH = "/opt/airflow/data/features.csv"
MODEL_PATH = "/opt/airflow/artifacts/prophet_preperiod.pkl"

default_args = {"owner": "you", "retries": 1, "retry_delay": timedelta(minutes=5)}

def train_prophet():
    df = pd.read_csv(DATA_PATH, parse_dates=["date"]).set_index("date")
    campaign_days = df[df["brand_burst_flag"]==1].index
    start = campaign_days.min()
    pre = df.loc[:start - pd.Timedelta(days=1)].reset_index().rename(columns={"date":"ds","y":"y"})
    regressors = [c for c in df.columns if c not in ["y","brand_burst_flag"]]
    m = Prophet(weekly_seasonality=True, daily_seasonality=False, interval_width=0.95)
    for r in regressors: m.add_regressor(r)
    m.fit(pre)
    with open(MODEL_PATH, "wb") as f:
        pickle.dump({"model": m, "regressors": regressors}, f)

with DAG(
    dag_id="retrain_weekly",
    default_args=default_args,
    schedule_interval="0 3 * * 0",  # Sundays 03:00
    start_date=datetime(2025, 1, 1),
    catchup=False,
    description="Weekly retrain Prophet pre-period model"
) as dag:
    train = PythonOperator(task_id="train_prophet", python_callable=train_prophet)
