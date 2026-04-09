# agent/tools.py
# ─────────────────────────────────────────────────────────────
# KairosAI AI Agent — Tool Definitions
# Three tools the agent can call:
#   1. sql_query    — query the data warehouse
#   2. vector_search — search document chunks
#   3. live_price   — get live prices from Alpaca
# ─────────────────────────────────────────────────────────────

import os
import re
from sqlalchemy import text
from database.db import engine
from rag.retriever import retrieve_for_agent
from agent.prompts import TOOL_DESCRIPTIONS
from dotenv import load_dotenv

load_dotenv()


# ── SQL TOOL ──────────────────────────────────────────────────

def sql_query(query: str) -> str:
    # Executes a read-only SQL query on the data warehouse
    # Returns results as a formatted string for the agent
    # Safety: only allows SELECT statements

    # Strip whitespace and normalize
    query = query.strip()

    # Remove markdown code blocks if agent wraps in them
    query = re.sub(r'```sql\s*', '', query)
    query = re.sub(r'```\s*', '', query)
    query = query.strip()

    # Safety check — only allow SELECT
    if not query.upper().startswith("SELECT"):
        return "Error: Only SELECT queries are allowed for safety."

    # Add LIMIT if not present to prevent huge results
    if "LIMIT" not in query.upper():
        query = query.rstrip(";") + " LIMIT 20"

    try:
        with engine.connect() as conn:
            result = conn.execute(text(query))
            rows   = result.fetchall()
            cols   = result.keys()

        if not rows:
            return "Query returned no results."

        # Format as readable table
        col_names = list(cols)
        lines     = [" | ".join(col_names)]
        lines.append("-" * len(lines[0]))

        for row in rows:
            line = " | ".join(
                str(round(v, 4)) if isinstance(v, float)
                else str(v)
                for v in row
            )
            lines.append(line)

        return "\n".join(lines)

    except Exception as e:
        return f"SQL Error: {str(e)}"


# ── VECTOR SEARCH TOOL ────────────────────────────────────────

def vector_search(query: str) -> str:
    # Searches the knowledge base for relevant context
    # Returns formatted chunks for the agent to use
    try:
        context = retrieve_for_agent(query)
        return context if context else "No relevant context found."
    except Exception as e:
        return f"Search Error: {str(e)}"


# ── LIVE PRICE TOOL ───────────────────────────────────────────

def live_price(input_str: str) -> str:
    # Gets live price data from Alpaca or latest from database
    # Input: ticker symbol or "portfolio"
    input_str = input_str.strip().upper()

    # ── Portfolio view ──
    if input_str == "PORTFOLIO":
        return get_portfolio()

    ticker = input_str

    # ── Try Alpaca API first ──
    try:
        import alpaca_trade_api as tradeapi

        api = tradeapi.REST(
            key_id     = os.getenv("ALPACA_API_KEY"),
            secret_key = os.getenv("ALPACA_SECRET_KEY"),
            base_url   = os.getenv("ALPACA_BASE_URL", "https://paper-api.alpaca.markets")
        )

        quote = api.get_latest_quote(ticker)
        price = float(quote.ap or quote.bp or 0)

        if price > 0:
            return f"{ticker}: ${price:.2f} (live from Alpaca)"

    except Exception:
        pass

    # ── Fall back to latest from live_quotes table ──
    try:
        with engine.connect() as conn:
            result = conn.execute(
                text("""
                    SELECT price, timestamp
                    FROM live_quotes
                    WHERE ticker = :ticker
                    ORDER BY timestamp DESC
                    LIMIT 1
                """),
                {"ticker": ticker}
            )
            row = result.fetchone()

        if row:
            return f"{ticker}: ${float(row.price):.2f} (as of {row.timestamp})"

    except Exception:
        pass

    # ── Fall back to latest close from fact_prices ──
    try:
        with engine.connect() as conn:
            result = conn.execute(
                text("""
                    SELECT close, date
                    FROM fact_prices
                    WHERE ticker = :ticker
                    ORDER BY date DESC
                    LIMIT 1
                """),
                {"ticker": ticker}
            )
            row = result.fetchone()

        if row:
            return f"{ticker}: ${float(row.close):.2f} (last close on {row.date})"

    except Exception as e:
        return f"Price Error: {str(e)}"

    return f"No price data found for {ticker}"


def get_portfolio() -> str:
    # Gets current paper portfolio from Alpaca
    try:
        import alpaca_trade_api as tradeapi

        api = tradeapi.REST(
            key_id     = os.getenv("ALPACA_API_KEY"),
            secret_key = os.getenv("ALPACA_SECRET_KEY"),
            base_url   = os.getenv("ALPACA_BASE_URL", "https://paper-api.alpaca.markets")
        )

        account   = api.get_account()
        positions = api.list_positions()

        lines = [
            f"Portfolio Value : ${float(account.portfolio_value):,.2f}",
            f"Cash            : ${float(account.cash):,.2f}",
            f"Buying Power    : ${float(account.buying_power):,.2f}",
            ""
        ]

        if positions:
            lines.append("Current Positions:")
            for pos in positions:
                pnl  = float(pos.unrealized_pl)
                sign = "+" if pnl >= 0 else ""
                lines.append(
                    f"  {pos.symbol:6s} {pos.qty:>6} shares @ "
                    f"${float(pos.avg_entry_price):.2f} | "
                    f"P&L: {sign}${pnl:.2f}"
                )
        else:
            lines.append("No open positions.")

        return "\n".join(lines)

    except Exception as e:
        return f"Portfolio Error: {str(e)}"


# ── LANGCHAIN TOOL WRAPPERS ───────────────────────────────────
# These wrap the functions above as LangChain Tool objects
# so the agent can call them automatically

from langchain.tools import Tool

SQL_TOOL = Tool(
    name        = "sql_query",
    func        = sql_query,
    description = TOOL_DESCRIPTIONS["sql_query"]
)

VECTOR_TOOL = Tool(
    name        = "vector_search",
    func        = vector_search,
    description = TOOL_DESCRIPTIONS["vector_search"]
)

PRICE_TOOL = Tool(
    name        = "live_price",
    func        = live_price,
    description = TOOL_DESCRIPTIONS["live_price"]
)

TOOLS = [SQL_TOOL, VECTOR_TOOL, PRICE_TOOL]