# dashboard/pages/sentiment.py
# ─────────────────────────────────────────────────────────────
# KairosAI Dashboard — Sentiment Analysis Page
# Shows sentiment heatmap across tickers and dates
# ─────────────────────────────────────────────────────────────
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import requests
from sqlalchemy import text
from database.db import engine

st.set_page_config(
    page_title = "Sentiment — KairosAI",
    layout     = "wide"
)

API_BASE = "http://localhost:5000"

TICKERS = [
    "AAPL", "MSFT", "NVDA", "GOOGL", "AMZN",
    "META", "TSLA", "SPY",  "QQQ",  "JPM",
    "BAC",  "UNH",  "JNJ",  "XOM",  "AMD"
]

if not st.session_state.get("logged_in"):
    st.error("Please sign in from the main page")
    st.stop()

st.title("Sentiment Analysis")
st.caption("FinBERT sentiment scores across tickers and time")

days = st.slider("Lookback days", 7, 60, 30)

# ── LOAD SENTIMENT DATA ───────────────────────────────────────
with engine.connect() as conn:
    result = conn.execute(
        text("""
            SELECT ticker, date, sentiment_score
            FROM fact_signals
            WHERE sentiment_score IS NOT NULL
            AND date >= CURRENT_DATE - :days * INTERVAL '1 day'
            ORDER BY date ASC
        """),
        {"days": days}
    )
    rows = result.fetchall()

if not rows:
    st.warning("No sentiment data available. Run models/sentiment.py first.")
    st.stop()

df = pd.DataFrame(rows, columns=["ticker", "date", "sentiment_score"])
df["date"]            = pd.to_datetime(df["date"])
df["sentiment_score"] = df["sentiment_score"].astype(float)

# ── HEATMAP ───────────────────────────────────────────────────
st.subheader("Sentiment heatmap")

pivot = df.pivot_table(
    index   = "ticker",
    columns = "date",
    values  = "sentiment_score",
    aggfunc = "mean"
)

if not pivot.empty:
    fig = go.Figure(data=go.Heatmap(
        z            = pivot.values,
        x            = [str(d.date()) for d in pivot.columns],
        y            = pivot.index.tolist(),
        colorscale   = "RdYlGn",
        zmid         = 0,
        colorbar     = dict(title="Sentiment"),
        hovertemplate = "Ticker: %{y}<br>Date: %{x}<br>Score: %{z:.4f}<extra></extra>"
    ))

    fig.update_layout(
        xaxis_title = "Date",
        yaxis_title = "Ticker",
        height      = 500
    )

    st.plotly_chart(fig, use_container_width=True)

# ── CURRENT SENTIMENT BAR ─────────────────────────────────────
st.subheader("Current sentiment by ticker")

latest = df.groupby("ticker")["sentiment_score"].mean().reset_index()
latest = latest.sort_values("sentiment_score", ascending=True)
latest["color"] = latest["sentiment_score"].apply(
    lambda x: "green" if x > 0.1 else "red" if x < -0.1 else "gray"
)

fig2 = go.Figure(go.Bar(
    x           = latest["sentiment_score"],
    y           = latest["ticker"],
    orientation = "h",
    marker_color = latest["color"]
))

fig2.add_vline(x=0, line_color="black", line_width=1)
fig2.update_layout(
    xaxis_title = "Sentiment score",
    yaxis_title = "Ticker",
    height      = 450
)

st.plotly_chart(fig2, use_container_width=True)

# ── SECTOR SENTIMENT ──────────────────────────────────────────
st.subheader("Sector sentiment")

try:
    with engine.connect() as conn:
        result = conn.execute(
            text("""
                SELECT sector, date, sentiment
                FROM sector_sentiment
                WHERE date >= CURRENT_DATE - :days * INTERVAL '1 day'
                ORDER BY date ASC
            """),
            {"days": days}
        )
        sector_rows = result.fetchall()

    if sector_rows:
        sector_df = pd.DataFrame(sector_rows, columns=["sector", "date", "sentiment"])
        sector_df["date"] = pd.to_datetime(sector_df["date"])

        fig3 = px.line(
            sector_df,
            x     = "date",
            y     = "sentiment",
            color = "sector",
            title = "Sector sentiment over time"
        )
        fig3.add_hline(y=0, line_dash="dash", line_color="gray")
        st.plotly_chart(fig3, use_container_width=True)

except Exception:
    st.info("Sector sentiment data not available")

# ── RECENT NEWS ───────────────────────────────────────────────
st.subheader("Recent news headlines")

selected_ticker = st.selectbox("Select ticker", TICKERS)

with engine.connect() as conn:
    result = conn.execute(
        text("""
            SELECT headline, source, published_at
            FROM fact_news
            WHERE ticker = :ticker
            ORDER BY published_at DESC
            LIMIT 10
        """),
        {"ticker": selected_ticker}
    )
    news_rows = result.fetchall()

if news_rows:
    for row in news_rows:
        st.markdown(f"**{row.headline}**")
        st.caption(f"{row.source} — {row.published_at}")
        st.divider()
else:
    st.info(f"No recent news for {selected_ticker}")