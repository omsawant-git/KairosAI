# dashboard/pages/portfolio.py
# ─────────────────────────────────────────────────────────────
# KairosAI Dashboard — Portfolio Page
# Shows live paper trading positions and P&L
# Auto-refreshes every 30 seconds
# ─────────────────────────────────────────────────────────────

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import requests
import time

st.set_page_config(
    page_title = "Portfolio — KairosAI",
    layout     = "wide"
)

API_BASE = "http://localhost:5000"


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


# ── AUTH CHECK ────────────────────────────────────────────────
if not st.session_state.get("logged_in"):
    st.error("Please sign in from the main page")
    st.stop()


# ── PAGE ──────────────────────────────────────────────────────
st.title("Portfolio")
st.caption("Live paper trading account — Alpaca")

# Auto refresh toggle
col1, col2, col3 = st.columns([6, 2, 1])
with col2:
    auto_refresh = st.toggle("Auto refresh", value=False)
with col3:
    if st.button("Refresh"):
        st.rerun()

# ── ACCOUNT SUMMARY ───────────────────────────────────────────
account = api_get("/portfolio/account")

if "error" not in account:
    col1, col2, col3, col4 = st.columns(4)
    col1.metric(
        "Portfolio Value",
        f"${account.get('portfolio_value', 0):,.2f}"
    )
    col2.metric(
        "Cash",
        f"${account.get('cash', 0):,.2f}"
    )
    col3.metric(
        "Buying Power",
        f"${account.get('buying_power', 0):,.2f}"
    )
    col4.metric(
        "Equity",
        f"${account.get('equity', 0):,.2f}"
    )
else:
    st.warning("Could not load account data")

st.divider()

# ── POSITIONS ─────────────────────────────────────────────────
st.subheader("Open Positions")

positions_data = api_get("/portfolio/positions")

if "error" not in positions_data:
    positions = positions_data.get("data", [])

    if positions:
        df = pd.DataFrame(positions)

        # Format columns
        df["unrealized_pl"]   = df["unrealized_pl"].apply(
            lambda x: f"+${x:.2f}" if x >= 0 else f"-${abs(x):.2f}"
        )
        df["unrealized_plpc"] = df["unrealized_plpc"].apply(
            lambda x: f"+{x:.2f}%" if x >= 0 else f"{x:.2f}%"
        )
        df["avg_entry_price"] = df["avg_entry_price"].apply(
            lambda x: f"${x:.2f}"
        )
        df["current_price"]   = df["current_price"].apply(
            lambda x: f"${x:.2f}"
        )
        df["market_value"]    = df["market_value"].apply(
            lambda x: f"${x:,.2f}"
        )

        st.dataframe(
            df[[
                "ticker", "qty", "avg_entry_price",
                "current_price", "market_value",
                "unrealized_pl", "unrealized_plpc", "side"
            ]],
            use_container_width=True,
            hide_index=True
        )

        # P&L chart
        positions_raw = api_get("/portfolio/positions").get("data", [])
        if positions_raw:
            fig = px.bar(
                pd.DataFrame(positions_raw),
                x     = "ticker",
                y     = "unrealized_pl",
                color = "unrealized_pl",
                color_continuous_scale = ["red", "green"],
                title = "Unrealized P&L by position"
            )
            fig.update_layout(showlegend=False)
            st.plotly_chart(fig, use_container_width=True)

    else:
        st.info("No open positions. Place your first trade on the Trade page.")
else:
    st.error("Could not load positions")

st.divider()

# ── LIVE PRICES ───────────────────────────────────────────────
st.subheader("Live Watchlist Prices")

prices_data = api_get("/portfolio/prices")

if "error" not in prices_data:
    prices = prices_data.get("data", [])

    if prices:
        df_prices = pd.DataFrame(prices)
        df_prices["price"] = df_prices["price"].apply(lambda x: f"${x:.2f}")

        cols = st.columns(5)
        for i, row in enumerate(prices[:15]):
            with cols[i % 5]:
                st.metric(
                    label = row["ticker"],
                    value = f"${float(row['price']):.2f}"
                )

# ── ORDER HISTORY ─────────────────────────────────────────────
st.divider()
st.subheader("Recent Orders")

orders_data = api_get("/portfolio/orders")

if "error" not in orders_data:
    orders = orders_data.get("data", [])

    if orders:
        df_orders = pd.DataFrame(orders)
        st.dataframe(
            df_orders[[
                "ticker", "qty", "side", "type",
                "status", "filled_avg_price", "filled_at"
            ]],
            use_container_width=True,
            hide_index=True
        )
    else:
        st.info("No orders yet.")

# ── AUTO REFRESH ──────────────────────────────────────────────
if auto_refresh:
    time.sleep(30)
    st.rerun()