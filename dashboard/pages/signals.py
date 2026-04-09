# dashboard/pages/signals.py
# ─────────────────────────────────────────────────────────────
# KairosAI Dashboard — ML Signals Page
# Shows forecast vs actual, anomaly timeline
# ─────────────────────────────────────────────────────────────
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import requests

st.set_page_config(
    page_title = "Signals — KairosAI",
    layout     = "wide"
)

API_BASE = "http://localhost:5000"

TICKERS = [
    "AAPL", "MSFT", "NVDA", "GOOGL", "AMZN",
    "META", "TSLA", "SPY",  "QQQ",  "JPM",
    "BAC",  "UNH",  "JNJ",  "XOM",  "AMD"
]


def api_get(endpoint: str) -> dict:
    try:
        response = requests.get(
            f"{API_BASE}{endpoint}",
            headers={"Authorization": f"Bearer {st.session_state.get('token')}"},
            timeout=10
        )
        return response.json()
    except Exception as e:
        return {"error": str(e)}


if not st.session_state.get("logged_in"):
    st.error("Please sign in from the main page")
    st.stop()

st.title("ML Signals")
st.caption("Forecasting, anomaly detection, and combined signals")

ticker = st.selectbox("Select ticker", TICKERS)
days   = st.slider("Lookback days", 30, 365, 90)

# ── FORECAST CHART ────────────────────────────────────────────
st.subheader(f"{ticker} — Forecast vs Actual")

from sqlalchemy import text
from database.db import engine

with engine.connect() as conn:
    result = conn.execute(
        text("""
            SELECT
                fp.date,
                fp.close AS actual,
                fs.predicted_close AS forecast,
                fs.is_anomaly,
                fs.anomaly_score
            FROM fact_prices fp
            LEFT JOIN fact_signals fs
                ON fp.ticker = fs.ticker
                AND fp.date  = fs.date
            WHERE fp.ticker = :ticker
            AND fp.date >= CURRENT_DATE - :days * INTERVAL '1 day'
            ORDER BY fp.date ASC
        """),
        {"ticker": ticker, "days": days}
    )
    rows = result.fetchall()

if rows:
    df = pd.DataFrame(rows, columns=[
        "date", "actual", "forecast", "is_anomaly", "anomaly_score"
    ])

    fig = go.Figure()

    # Actual price line
    fig.add_trace(go.Scatter(
        x    = df["date"],
        y    = df["actual"],
        name = "Actual price",
        line = dict(color="royalblue", width=2)
    ))

    # Forecast line
    forecast_df = df[df["forecast"].notna()]
    if not forecast_df.empty:
        fig.add_trace(go.Scatter(
            x    = forecast_df["date"],
            y    = forecast_df["forecast"],
            name = "Forecast",
            line = dict(color="orange", width=1.5, dash="dash")
        ))

    # Anomaly markers
    anomaly_df = df[df["is_anomaly"] == True]
    if not anomaly_df.empty:
        fig.add_trace(go.Scatter(
            x      = anomaly_df["date"],
            y      = anomaly_df["actual"],
            mode   = "markers",
            name   = "Anomaly",
            marker = dict(color="red", size=10, symbol="x")
        ))

    fig.update_layout(
        xaxis_title = "Date",
        yaxis_title = "Price ($)",
        height      = 450,
        hovermode   = "x unified"
    )

    st.plotly_chart(fig, use_container_width=True)

    # ── ANOMALY SCORE CHART ───────────────────────────────────
    st.subheader(f"{ticker} — Anomaly scores")

    score_df = df[df["anomaly_score"].notna()]
    if not score_df.empty:
        fig2 = go.Figure()
        fig2.add_trace(go.Bar(
            x     = score_df["date"],
            y     = score_df["anomaly_score"],
            name  = "Anomaly score",
            marker_color = score_df["anomaly_score"].apply(
                lambda x: "red" if float(x) > 0.6 else "orange" if float(x) > 0.4 else "steelblue"
            )
        ))
        fig2.add_hline(
            y=0.6, line_dash="dash",
            line_color="red",
            annotation_text="High anomaly threshold"
        )
        fig2.update_layout(
            xaxis_title = "Date",
            yaxis_title = "Anomaly score",
            height      = 300
        )
        st.plotly_chart(fig2, use_container_width=True)
    else:
        st.info("No anomaly scores available for this period")

    # ── METRICS ───────────────────────────────────────────────
    st.divider()
    col1, col2, col3, col4 = st.columns(4)

    n_anomalies = int(df["is_anomaly"].sum()) if "is_anomaly" in df.columns else 0
    col1.metric("Total days",    len(df))
    col2.metric("Anomalies",     n_anomalies)
    col3.metric("Anomaly rate",  f"{n_anomalies/len(df)*100:.1f}%")

    if not forecast_df.empty and len(forecast_df) > 1:
        actuals = forecast_df["actual"].values
        preds   = forecast_df["forecast"].values
        import numpy as np
        mape = float(np.mean(np.abs((actuals - preds) / actuals)) * 100)
        col4.metric("Forecast MAPE", f"{mape:.2f}%")

else:
    st.info("No data available for this ticker and period")