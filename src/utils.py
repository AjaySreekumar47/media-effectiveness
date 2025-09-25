import pandas as pd
from prophet import Prophet

def load_features(path="data/features.csv"):
    df = pd.read_csv(path, parse_dates=["date"]).set_index("date")
    return df

def make_regressor_list(df):
    return [c for c in df.columns if c not in ["y", "brand_burst_flag"]]

def fit_prophet_preperiod(df_feat, campaign_start):
    regressors = make_regressor_list(df_feat)
    pre_df = df_feat.loc[:campaign_start - pd.Timedelta(days=1)].reset_index().rename(columns={"date":"ds","y":"y"})
    m = Prophet(weekly_seasonality=True, daily_seasonality=False, interval_width=0.95)
    for r in regressors:
        m.add_regressor(r)
    m.fit(pre_df)
    return m, regressors

def prophet_counterfactual(m, df_feat, regressors, start, end):
    post = df_feat.loc[start:end].reset_index().rename(columns={"date":"ds","y":"y"})
    future = post[["ds"] + regressors]
    fc = m.predict(future)
    out = post[["ds","y"]].copy()
    out["yhat"] = fc["yhat"].values
    out["yhat_lower"] = fc["yhat_lower"].values
    out["yhat_upper"] = fc["yhat_upper"].values
    return out

def compute_lift(counterfactual_df):
    lift_series = counterfactual_df["y"] - counterfactual_df["yhat"]
    cum = float(lift_series.sum())
    rel = float((lift_series.sum() / counterfactual_df["yhat"].sum())*100)
    return cum, rel
