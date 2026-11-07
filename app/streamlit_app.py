"""Bay Area Traffic Speed Forecaster - run with:  streamlit run app/streamlit_app.py"""
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from traffic.inference import load_context, forecast, sensor_metrics

st.set_page_config(page_title="Bay Area Traffic Forecaster", layout="wide")
st.title("🚗 Bay Area Traffic Speed Forecaster")


@st.cache_resource(show_spinner="Loading data and models...")
def get_ctx(data_dir):
    return load_context(data_dir, str(ROOT / "models"))


@st.cache_data(show_spinner="Computing test-set errors per sensor...")
def get_sensor_metrics(data_dir):
    return sensor_metrics(get_ctx(data_dir))


data_dir = st.sidebar.text_input("Data directory", os.environ.get("TRAFFIC_DATA_DIR", str(ROOT / "data/raw")))
try:
    ctx = get_ctx(data_dir)
except FileNotFoundError as e:
    st.error(f"{e}\n\nSee README: put the data in the data directory and run `python scripts/train_lgbm.py`.")
    st.stop()

ids = list(ctx.clean.columns)
sensor = st.sidebar.selectbox("Sensor", ids)
si = ids.index(sensor)

# Forecast origin: only the held-out TEST period (the models never saw it in training)
idx = ctx.clean.index
lo = idx[ctx.val_end + ctx.cfg.max_lag].to_pydatetime()
hi = idx[ctx.test_end - 1 - max(ctx.cfg.horizons)].to_pydatetime()
now = st.sidebar.slider("Forecast made at (test period)", lo, hi, lo.replace(hour=7, minute=30) if lo <= lo.replace(hour=7, minute=30) <= hi else lo,
                        step=pd.Timedelta(minutes=5).to_pytimedelta(), format="MMM D, HH:mm")
hist_h = st.sidebar.slider("History shown (hours)", 1, 12, 3)
t = int(idx.get_loc(pd.Timestamp(now)))

pred = forecast(ctx, t).loc[sensor]                       # minutes ahead -> mph
fut_t = [idx[t + h // 5] for h in pred.index]
actual = [ctx.raw.iloc[t + h // 5, si] for h in pred.index]
actual = [a if a > 0 else float("nan") for a in actual]
hist = ctx.raw.iloc[max(0, t - hist_h * 12): t + 1, si].where(lambda s: s > 0)

fig = go.Figure()
fig.add_scatter(x=hist.index, y=hist.values, name="Observed (history)", line=dict(color="#1f77b4"))
fig.add_scatter(x=[idx[t]] + fut_t, y=[ctx.clean.iloc[t, si]] + list(pred.values), name="Forecast",
                mode="lines+markers", line=dict(color="#d62728", dash="dash"))
fig.add_scatter(x=fut_t, y=actual, name="Actual (what really happened)", mode="markers",
                marker=dict(color="#2ca02c", size=9, symbol="diamond"))
fig.add_vline(x=idx[t], line_dash="dot", line_color="gray")
fig.update_layout(height=420, yaxis_title="Speed (mph)", margin=dict(t=20), legend=dict(orientation="h"))
st.plotly_chart(fig, width="stretch")

table = pd.DataFrame({"Horizon (min)": pred.index, "Predicted (mph)": pred.values.round(1),
                      "Actual (mph)": [round(float(a), 1) for a in actual]})
table["Abs. error"] = (table["Predicted (mph)"] - table["Actual (mph)"]).abs().round(1)

c1, c2 = st.columns(2)
with c1:
    st.subheader("This forecast")
    st.dataframe(table, hide_index=True, width="stretch")
    loc = ctx.sensors.loc[sensor]
    st.caption(f"Sensor {sensor} at ({loc.latitude:.4f}, {loc.longitude:.4f})")
    m = pd.DataFrame({"lat": ctx.sensors.latitude, "lon": ctx.sensors.longitude})
    m["color"], m["size"] = "#9aa0a6", 30
    m.loc[sensor, ["color", "size"]] = ["#d62728", 120]
    st.map(m.dropna(), latitude="lat", longitude="lon", color="color", size="size", height=260)

with c2:
    st.subheader(f"Test-set error for sensor {sensor}")
    sm = get_sensor_metrics(data_dir)
    mine = sm[sm.sensor_id == sensor].set_index("horizon_min")[["MAE", "RMSE", "MAPE", "persistence_MAE"]]
    st.dataframe(mine.round(2), width="stretch")
    st.caption("MAE/RMSE in mph, MAPE in %. `persistence_MAE` = error if you just assume speed stays the same.")
    st.subheader("Whole network")
    rep = ROOT / "reports" / "metrics.csv"
    if rep.exists():
        g = pd.read_csv(rep)
        st.dataframe(g.pivot(index="horizon_min", columns="model", values="MAE").round(2),
                     width="stretch")
        st.caption("MAE (mph) per horizon, from reports/metrics.csv")
    else:
        st.info("Run scripts/train_lgbm.py to generate reports/metrics.csv")
