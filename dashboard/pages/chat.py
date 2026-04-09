# dashboard/pages/chat.py
# ─────────────────────────────────────────────────────────────
# KairosAI Dashboard — AI Agent Chat Page
# Interactive chat with the KairosAI financial analyst agent
# Shows tool trace for transparency
# ─────────────────────────────────────────────────────────────

import streamlit as st
import requests

st.set_page_config(
    page_title = "AI Agent — KairosAI",
    layout     = "wide"
)

API_BASE = "http://localhost:5000"


def api_post(endpoint: str, data: dict) -> dict:
    try:
        response = requests.post(
            f"{API_BASE}{endpoint}",
            json=data,
            headers={"Authorization": f"Bearer {st.session_state.get('token')}"},
            timeout=120
        )
        return response.json()
    except Exception as e:
        return {"error": str(e)}


if not st.session_state.get("logged_in"):
    st.error("Please sign in from the main page")
    st.stop()

st.title("AI Agent")
st.caption("Ask anything about the market — grounded in real data")

# ── SESSION STATE ─────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []

# ── SIDEBAR CONTROLS ──────────────────────────────────────────
with st.sidebar:
    st.subheader("Agent controls")

    if st.button("Clear conversation", use_container_width=True):
        st.session_state.messages = []
        api_post("/agent/reset", {})
        st.rerun()

    st.divider()
    st.caption("Available tools")
    st.markdown("- **sql_query** — data warehouse")
    st.markdown("- **vector_search** — news & signals")
    st.markdown("- **live_price** — Alpaca prices")

    st.divider()
    st.caption("Example questions")
    examples = [
        "What is the sentiment for NVDA?",
        "Are there anomalies in TSLA recently?",
        "Compare AAPL and META sentiment",
        "What is my current portfolio?",
        "Which tickers have the highest anomaly scores?",
        "What does the macro data say about inflation?"
    ]
    for example in examples:
        if st.button(example, use_container_width=True):
            st.session_state.messages.append({
                "role":    "user",
                "content": example
            })
            st.rerun()

# ── CHAT HISTORY ──────────────────────────────────────────────
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

        # Show tool trace for assistant messages
        if message["role"] == "assistant" and message.get("trace"):
            with st.expander(f"Tool trace ({len(message['trace'])} steps)"):
                for step in message["trace"]:
                    st.markdown(f"**Tool:** `{step['tool']}`")
                    st.markdown(f"**Input:** {step['input']}")
                    st.markdown(f"**Result:** {step['observation'][:200]}...")
                    st.divider()

# ── CHAT INPUT ────────────────────────────────────────────────
if prompt := st.chat_input("Ask about the market..."):
    # Add user message
    st.session_state.messages.append({
        "role":    "user",
        "content": prompt
    })

    with st.chat_message("user"):
        st.markdown(prompt)

    # Get agent response
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            result = api_post("/agent/chat", {"message": prompt})

        if "error" in result:
            response = f"Error: {result['error']}"
            trace    = []
        else:
            response = result.get("response", "No response")
            trace    = result.get("trace", [])

        st.markdown(response)

        # Show tool trace
        if trace:
            with st.expander(f"Tool trace ({len(trace)} steps)"):
                for step in trace:
                    st.markdown(f"**Tool:** `{step['tool']}`")
                    st.markdown(f"**Input:** {step['input']}")
                    st.markdown(f"**Result:** {step['observation'][:200]}...")
                    st.divider()

    # Store assistant message
    st.session_state.messages.append({
        "role":    "assistant",
        "content": response,
        "trace":   trace
    })