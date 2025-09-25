import pandas as pd
from prophet import Prophet
from .utils import make_regressor_list

def run_scenario(df_feat, campaign_start, campaign_end, channel, multiplier, m_pre=None):
    df = df_feat.copy()
    df.loc[campaign_start:campaign_end, channel] = df.loc[campaign_start:campaign_end, channel] * multiplier
    regressors = make_regressor_list(df)
    if m_pre is None:
        pre_df = df.loc[:campaign_start - pd.Timedelta(days=1)].reset_index().rename(columns={"date":"ds","y":"y"})
        m_pre = Prophet(weekly_seasonality=True, daily_seasonality=False, interval_width=0.95)
        for r in regressors:
            m_pre.add_regressor(r)
        m_pre.fit(pre_df)
    post_df = df.loc[campaign_start:campaign_end].reset_index().rename(columns={"date":"ds","y":"y"})
    future = post_df[["ds"] + regressors]
    fc = m_pre.predict(future)
    out = post_df.assign(yhat=fc["yhat"].values, yhat_lower=fc["yhat_lower"].values, yhat_upper=fc["yhat_upper"].values)
    return out
