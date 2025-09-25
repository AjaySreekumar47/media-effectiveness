import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from src.utils import load_features, fit_prophet_preperiod, prophet_counterfactual, compute_lift
from src.scenario import run_scenario

st.set_page_config(page_title="Media Effectiveness Suite", layout="wide")
st.sidebar.header("Data & Campaign")

# Load data
df_feat = load_features("data/features.csv")
campaign_days = df_feat[df_feat["brand_burst_flag"] == 1].index
campaign_start, campaign_end = campaign_days.min(), campaign_days.max()
st.sidebar.write(f"**Campaign:** {campaign_start.date()} → {campaign_end.date()}")

@st.cache_resource
def _fit_pre(df, start):
    return fit_prophet_preperiod(df, start)

m_pre, regressors = _fit_pre(df_feat, campaign_start)
baseline = prophet_counterfactual(m_pre, df_feat, regressors, campaign_start, campaign_end)
cum_lift, rel_lift = compute_lift(baseline)

tab1, tab2, tab3, tab4, tab5 = st.tabs(
    ["Overview", "Forecasts", "Causal Impact", "Scenario Simulator", "Attribution (Notes)"]
)

with tab1:
    st.markdown("### Project Overview")
    st.write("""- **Phase 2:** Forecasting (Prophet best)
- **Phase 3:** Causal impact (Prophet counterfactual) → Lift & iROAS
- **Phase 4:** ML attribution (Lasso, XGB+SHAP)
- **Phase 5:** Unified reporting""")
    col1, col2 = st.columns(2)
    col1.metric("Campaign lift (cumulative)", f"{cum_lift:,.1f}")
    col2.metric("Relative lift", f"{rel_lift:.2f}%")

with tab2:
    st.markdown("### Prophet Baseline Forecast (Campaign Window)")
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=baseline["ds"], y=baseline["y"], name="Actual", line=dict(color="black")))
    fig.add_trace(go.Scatter(x=baseline["ds"], y=baseline["yhat"], name="Counterfactual (yhat)", line=dict(color="red")))
    fig.add_trace(go.Scatter(x=baseline["ds"], y=baseline["yhat_upper"], name="95% Upper", line=dict(width=0), showlegend=False))
    fig.add_trace(go.Scatter(x=baseline["ds"], y=baseline["yhat_lower"], name="95% Lower",
                             fill="tonexty", mode="lines", line=dict(width=0), fillcolor="rgba(255,0,0,0.15)", showlegend=False))
    fig.add_vrect(x0=campaign_start, x1=campaign_end, fillcolor="orange", opacity=0.15, line_width=0)
    st.plotly_chart(fig, use_container_width=True)

with tab3:
    st.markdown("### Causal Summary")
    st.dataframe(pd.DataFrame({
        "Campaign": ["Jul 2022"],
        "Cumulative lift": [cum_lift],
        "Relative lift (%)": [rel_lift]
    }))
    iROAS = 0.17  # from your Phase 3 result
    fig2 = go.Figure()
    fig2.add_trace(go.Bar(x=["Campaign Jul 2022"], y=[iROAS], marker_color="darkorange", name="iROAS"))
    fig2.add_hline(y=1.0, line_dash="dash", line_color="red", annotation_text="Break-even (1.0)")
    st.plotly_chart(fig2, use_container_width=True)

with tab4:
    st.markdown("### Scenario Simulator")
    chan_options = [c for c in regressors if "brand_burst" not in c]
    channel = st.selectbox("Channel to change", options=chan_options, index=chan_options.index("search_spend") if "search_spend" in chan_options else 0)
    pct = st.slider("Multiplier (%)", 50, 200, 120, 5)
    if st.button("Run Scenario"):
        scen = run_scenario(df_feat, campaign_start, campaign_end, channel, pct/100.0, m_pre=m_pre)
        diff = (scen["yhat"].values - baseline["yhat"].values).sum()
        st.metric("Cumulative incremental vs baseline", f"{diff:,.1f}")
        fig3 = go.Figure()
        fig3.add_trace(go.Scatter(x=baseline["ds"], y=baseline["yhat"], name="Baseline yhat", line=dict(color="red")))
        fig3.add_trace(go.Scatter(x=scen["ds"], y=scen["yhat"], name=f"Scenario yhat ({channel} x{pct/100:.2f})", line=dict(color="green")))
        fig3.add_vrect(x0=campaign_start, x1=campaign_end, fillcolor="orange", opacity=0.15, line_width=0)
        st.plotly_chart(fig3, use_container_width=True)

with tab5:
    st.markdown("### Phase 4 Notes")
    st.write("""- Time features (rolling means & weekly seasonality) dominated predictions.
- Social spend was the most consistent media driver in SHAP.
- Lasso suggested display+social; XGB was weak as a forecaster in this setup.""")
