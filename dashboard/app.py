# dashboard/app.py
# ─────────────────────────────────────────────────────────────
# KairosAI Streamlit Dashboard — Main Entry Point
# ─────────────────────────────────────────────────────────────

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import streamlit as st
import requests
import pandas as pd
import plotly.express as px

# ── PAGE CONFIG ───────────────────────────────────────────────
st.set_page_config(
    page_title            = "KairosAI",
    page_icon             = "📊",
    layout                = "wide",
    initial_sidebar_state = "expanded"
)

API_BASE = "http://localhost:5000"

# ── SESSION STATE ─────────────────────────────────────────────
if "token"     not in st.session_state:
    st.session_state.token     = None
if "user"      not in st.session_state:
    st.session_state.user      = None
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False


# ── HELPERS ───────────────────────────────────────────────────

def api_get(endpoint: str) -> dict:
    try:
        response = requests.get(
            f"{API_BASE}{endpoint}",
            headers={"Authorization": f"Bearer {st.session_state.token}"},
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
            headers={"Authorization": f"Bearer {st.session_state.token}"},
            timeout=30
        )
        return response.json()
    except Exception as e:
        return {"error": str(e)}


def login(email: str, password: str) -> bool:
    try:
        response = requests.post(
            f"{API_BASE}/auth/login",
            json={"email": email, "password": password},
            timeout=10
        )
        data = response.json()
        if "access_token" in data:
            st.session_state.token     = data["access_token"]
            st.session_state.user      = data["user"]
            st.session_state.logged_in = True
            return True
        return False
    except Exception:
        return False


def logout():
    st.session_state.token     = None
    st.session_state.user      = None
    st.session_state.logged_in = False
    st.rerun()


# ── AUTH GATE ─────────────────────────────────────────────────

def show_login():
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.title("KairosAI")
        st.caption("AI-powered financial intelligence platform")
        st.divider()

        tab1, tab2 = st.tabs(["Login", "Register"])

        with tab1:
            st.subheader("Sign in")
            email    = st.text_input("Email",    key="login_email")
            password = st.text_input("Password", type="password", key="login_pass")

            if st.button("Sign in", use_container_width=True, type="primary"):
                if login(email, password):
                    st.success("Welcome back!")
                    st.rerun()
                else:
                    st.error("Invalid email or password")

        with tab2:
            st.subheader("Create account")
            reg_name  = st.text_input("Full name", key="reg_name")
            reg_email = st.text_input("Email",     key="reg_email")
            reg_pass  = st.text_input("Password",  type="password", key="reg_pass")

            if st.button("Create account", use_container_width=True, type="primary"):
                try:
                    response = requests.post(
                        f"{API_BASE}/auth/register",
                        json={
                            "email":     reg_email,
                            "password":  reg_pass,
                            "full_name": reg_name
                        },
                        timeout=10
                    )
                    data = response.json()
                    if "message" in data:
                        st.success("Account created. Please sign in.")
                    else:
                        st.error(data.get("error", "Registration failed"))
                except Exception as e:
                    st.error(str(e))


# ── SIDEBAR ───────────────────────────────────────────────────

def show_sidebar():
    with st.sidebar:
        st.title("KairosAI")
        st.caption(f"Welcome, {st.session_state.user.get('full_name', 'Trader')}")
        st.divider()

        st.markdown("### Navigation")
        st.markdown("- 📊 **app** — Overview")
        st.markdown("- 💼 **portfolio** — Portfolio")
        st.markdown("- 📈 **trade** — Trade")
        st.markdown("- 🔔 **signals** — Signals")
        st.markdown("- 📰 **sentiment** — Sentiment")
        st.markdown("- 🤖 **chat** — AI Agent")

        st.divider()
        if st.button("Sign out", use_container_width=True):
            logout()


# ── OVERVIEW PAGE ─────────────────────────────────────────────

def show_overview():
    st.title("Market Overview")
    st.caption("Live signals for all 15 tickers")

    col1, col2 = st.columns([8, 1])
    with col2:
        if st.button("🔄 Refresh"):
            st.rerun()

    data = api_get("/signals/summary/all")

    if "error" in data:
        st.error(f"API error: {data['error']}")
        st.info("Make sure Flask API is running: python -m api.app")
        return

    rows = data.get("data", [])

    if not rows:
        st.warning("No signal data available. Run python -m models.train first.")
        return

    # ── Metric cards ──
    col1, col2, col3, col4 = st.columns(4)

    bullish   = sum(1 for r in rows if r.get("direction") == "UP")
    bearish   = sum(1 for r in rows if r.get("direction") == "DOWN")
    anomalies = sum(1 for r in rows if r.get("is_anomaly"))
    sent_vals = [
        float(r["sentiment_score"])
        for r in rows if r.get("sentiment_score") is not None
    ]
    avg_sent = sum(sent_vals) / len(sent_vals) if sent_vals else 0

    col1.metric("Bullish signals",  f"{bullish}/15")
    col2.metric("Bearish signals",  f"{bearish}/15")
    col3.metric("Anomalies today",  anomalies)
    col4.metric("Avg sentiment",    f"{avg_sent:+.3f}")

    st.divider()

    # ── Signals table ──
    df = pd.DataFrame(rows)

    display_cols = [
        "ticker", "company", "sector",
        "current_price", "forecast_price",
        "direction", "change_pct",
        "is_anomaly", "sentiment_score"
    ]
    existing = [c for c in display_cols if c in df.columns]
    st.dataframe(df[existing], use_container_width=True, hide_index=True)

    st.divider()

    # ── Forecast chart ──
    st.subheader("Forecast direction by ticker")

    try:
        if "change_pct" in df.columns:
            chart_df = df[df["change_pct"].notna()].copy()
            if not chart_df.empty:
                chart_df["change_pct"] = chart_df["change_pct"].astype(float)
                chart_df = chart_df.sort_values("change_pct")
                fig = px.bar(
                    chart_df,
                    x     = "ticker",
                    y     = "change_pct",
                    color = "change_pct",
                    color_continuous_scale    = ["red", "gray", "green"],
                    color_continuous_midpoint = 0,
                    labels = {"change_pct": "Forecast change %"},
                    title  = "5-day forecast % change by ticker"
                )
                fig.add_hline(y=0, line_color="white", line_width=1)
                fig.update_layout(showlegend=False, height=400)
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("Forecast data will appear after models run.")
        else:
            st.info("Forecast data will appear after models run.")
    except Exception as e:
        st.info(f"Forecast chart not available: {e}")

    # ── Sentiment chart ──
    st.subheader("Sentiment by ticker")

    try:
        if "sentiment_score" in df.columns:
            sent_df = df[df["sentiment_score"].notna()].copy()
            if not sent_df.empty:
                sent_df["sentiment_score"] = sent_df["sentiment_score"].astype(float)
                sent_df = sent_df.sort_values("sentiment_score")
                sent_df["color"] = sent_df["sentiment_score"].apply(
                    lambda x: "green" if x > 0.1 else "red" if x < -0.1 else "gray"
                )
                fig2 = px.bar(
                    sent_df,
                    x     = "ticker",
                    y     = "sentiment_score",
                    color = "sentiment_score",
                    color_continuous_scale    = ["red", "gray", "green"],
                    color_continuous_midpoint = 0,
                    title = "Current sentiment score by ticker"
                )
                fig2.add_hline(y=0, line_color="white", line_width=1)
                fig2.update_layout(showlegend=False, height=350)
                st.plotly_chart(fig2, use_container_width=True)
    except Exception as e:
        st.info(f"Sentiment chart not available: {e}")


# ── MAIN ──────────────────────────────────────────────────────

def main():
    if not st.session_state.logged_in:
        show_login()
    else:
        show_sidebar()
        show_overview()


main()