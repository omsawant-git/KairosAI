# dashboard/pages/trade.py
# ─────────────────────────────────────────────────────────────
# KairosAI Dashboard — Trade Execution Page
# Place paper trades with ML signal preview
# ─────────────────────────────────────────────────────────────

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import requests

st.set_page_config(
    page_title = "Trade — KairosAI",
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


def api_post(endpoint: str, data: dict) -> dict:
    try:
        response = requests.post(
            f"{API_BASE}{endpoint}",
            json=data,
            headers={"Authorization": f"Bearer {st.session_state.get('token')}"},
            timeout=30
        )
        return response.json()
    except Exception as e:
        return {"error": str(e)}


# ── AUTH CHECK ────────────────────────────────────────────────
if not st.session_state.get("logged_in"):
    st.error("Please sign in from the main page")
    st.stop()


# ── PAGE ──────────────────────────────────────────────────────
st.title("Trade")
st.caption("AI-assisted paper trading — no real money")

col_left, col_right = st.columns([1, 2])

# ── LEFT: ORDER FORM ──────────────────────────────────────────
with col_left:
    st.subheader("Place order")

    ticker = st.selectbox("Ticker", TICKERS)
    side   = st.radio("Side", ["Buy", "Sell"], horizontal=True)
    qty    = st.number_input("Quantity", min_value=1, max_value=1000, value=10)

    # Fetch signal for selected ticker
    signal = api_get(f"/signals/{ticker}/latest")

    if "error" not in signal:
        st.divider()
        st.caption("AI signal preview")

        price = signal.get("current_price")
        if price:
            st.metric("Current price", f"${float(price):.2f}")

        direction = signal.get("direction")
        change    = signal.get("change_pct")
        if direction and change:
            color = "green" if direction == "UP" else "red"
            st.markdown(
                f"**Forecast:** :{color}[{direction} {abs(change):.2f}%]"
            )

        sentiment = signal.get("sentiment_score")
        if sentiment is not None:
            label = (
                "Positive" if float(sentiment) > 0.1
                else "Negative" if float(sentiment) < -0.1
                else "Neutral"
            )
            st.metric("Sentiment", f"{label} ({float(sentiment):+.3f})")

        is_anomaly = signal.get("is_anomaly")
        if is_anomaly:
            st.warning("Anomaly detected on this ticker")

    st.divider()

    # Estimated cost
    price_val = signal.get("current_price") if "error" not in signal else None
    if price_val:
        est_cost = float(price_val) * qty
        st.metric(
            "Estimated cost" if side == "Buy" else "Estimated proceeds",
            f"${est_cost:,.2f}"
        )

    # Submit button
    if st.button(
        f"Place {side} order — {qty} {ticker}",
        type="primary",
        use_container_width=True
    ):
        result = api_post("/portfolio/order", {
            "ticker": ticker,
            "qty":    qty,
            "side":   side.lower()
        })

        if "error" in result:
            st.error(f"Order failed: {result['error']}")
        else:
            st.success(
                f"Order placed — {result.get('status', 'submitted')}\n"
                f"Order ID: {result.get('order_id', '')}"
            )

# ── RIGHT: PRICE CHART ────────────────────────────────────────
with col_right:
    st.subheader(f"{ticker} — Price history")

    # Load price history
    from sqlalchemy import text
    from database.db import engine

    with engine.connect() as conn:
        result = conn.execute(
            text("""
                SELECT date, open, high, low, close, volume
                FROM fact_prices
                WHERE ticker = :ticker
                ORDER BY date DESC
                LIMIT 120
            """),
            {"ticker": ticker}
        )
        rows = result.fetchall()

    if rows:
        df = pd.DataFrame(rows, columns=["date", "open", "high", "low", "close", "volume"])
        df = df.sort_values("date")

        # Candlestick chart
        fig = go.Figure(data=[
            go.Candlestick(
                x     = df["date"],
                open  = df["open"],
                high  = df["high"],
                low   = df["low"],
                close = df["close"],
                name  = ticker
            )
        ])

        # Add forecast line if available
        if "error" not in signal and signal.get("forecast_price"):
            fig.add_hline(
                y          = float(signal["forecast_price"]),
                line_dash  = "dash",
                line_color = "orange",
                annotation_text = f"Forecast ${float(signal['forecast_price']):.2f}"
            )

        fig.update_layout(
            title          = f"{ticker} — Last 120 trading days",
            xaxis_title    = "Date",
            yaxis_title    = "Price ($)",
            xaxis_rangeslider_visible = False,
            height         = 500
        )

        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No price data available")