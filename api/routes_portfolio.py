# api/routes_portfolio.py
# ─────────────────────────────────────────────────────────────
# KairosAI Flask API — Portfolio & Trading Routes
# Handles Alpaca paper trading operations
# Endpoints:
#   GET  /portfolio/positions    — current holdings
#   GET  /portfolio/account      — account summary
#   POST /portfolio/order        — place a paper trade
#   GET  /portfolio/orders       — order history
#   GET  /portfolio/prices       — live quotes for watchlist
# ─────────────────────────────────────────────────────────────

import os
from flask import Blueprint, jsonify, request
from flask_jwt_extended import jwt_required
from dotenv import load_dotenv
from sqlalchemy import text
from database.db import engine

load_dotenv()

portfolio_bp = Blueprint("portfolio", __name__, url_prefix="/portfolio")


def get_alpaca_api():
    # Creates and returns an Alpaca API client
    import alpaca_trade_api as tradeapi
    return tradeapi.REST(
        key_id     = os.getenv("ALPACA_API_KEY"),
        secret_key = os.getenv("ALPACA_SECRET_KEY"),
        base_url   = os.getenv("ALPACA_BASE_URL", "https://paper-api.alpaca.markets")
    )


# ── ACCOUNT SUMMARY ───────────────────────────────────────────
@portfolio_bp.route("/account", methods=["GET"])
@jwt_required()
def get_account():
    # Returns paper trading account summary
    try:
        api     = get_alpaca_api()
        account = api.get_account()

        return jsonify({
            "portfolio_value": float(account.portfolio_value),
            "cash":            float(account.cash),
            "buying_power":    float(account.buying_power),
            "equity":          float(account.equity),
            "status":          account.status
        }), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ── POSITIONS ─────────────────────────────────────────────────
@portfolio_bp.route("/positions", methods=["GET"])
@jwt_required()
def get_positions():
    # Returns all current open positions
    try:
        api       = get_alpaca_api()
        positions = api.list_positions()

        data = []
        for pos in positions:
            data.append({
                "ticker":          pos.symbol,
                "qty":             float(pos.qty),
                "avg_entry_price": float(pos.avg_entry_price),
                "current_price":   float(pos.current_price),
                "market_value":    float(pos.market_value),
                "unrealized_pl":   float(pos.unrealized_pl),
                "unrealized_plpc": float(pos.unrealized_plpc) * 100,
                "side":            pos.side
            })

        return jsonify({
            "count": len(data),
            "data":  data
        }), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ── PLACE ORDER ───────────────────────────────────────────────
@portfolio_bp.route("/order", methods=["POST"])
@jwt_required()
def place_order():
    # Places a paper trade order via Alpaca
    # Expects JSON: { "ticker": "AAPL", "qty": 10, "side": "buy" }
    data   = request.get_json()
    ticker = data.get("ticker", "").upper()
    qty    = data.get("qty", 0)
    side   = data.get("side", "buy").lower()

    # Validate inputs
    if not ticker:
        return jsonify({"error": "ticker is required"}), 400

    if qty <= 0:
        return jsonify({"error": "qty must be greater than 0"}), 400

    if side not in ["buy", "sell"]:
        return jsonify({"error": "side must be buy or sell"}), 400

    try:
        api   = get_alpaca_api()
        order = api.submit_order(
            symbol        = ticker,
            qty           = qty,
            side          = side,
            type          = "market",
            time_in_force = "day"
        )

        return jsonify({
            "order_id":  str(order.id),
            "ticker":    order.symbol,
            "qty":       float(order.qty),
            "side":      order.side,
            "type":      order.type,
            "status":    order.status,
            "submitted": str(order.submitted_at)
        }), 201

    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ── ORDER HISTORY ─────────────────────────────────────────────
@portfolio_bp.route("/orders", methods=["GET"])
@jwt_required()
def get_orders():
    # Returns recent order history
    try:
        api    = get_alpaca_api()
        orders = api.list_orders(status="all", limit=20)

        data = []
        for order in orders:
            data.append({
                "order_id":  str(order.id),
                "ticker":    order.symbol,
                "qty":       float(order.qty),
                "side":      order.side,
                "type":      order.type,
                "status":    order.status,
                "filled_at": str(order.filled_at) if order.filled_at else None,
                "filled_avg_price": float(order.filled_avg_price) if order.filled_avg_price else None
            })

        return jsonify({
            "count": len(data),
            "data":  data
        }), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ── LIVE PRICES ───────────────────────────────────────────────
@portfolio_bp.route("/prices", methods=["GET"])
@jwt_required()
def get_live_prices():
    # Returns latest price for all watchlist tickers
    # Pulls from live_quotes table first, falls back to fact_prices
    try:
        with engine.connect() as conn:
            result = conn.execute(text("""
                SELECT DISTINCT ON (ticker)
                    ticker,
                    price,
                    timestamp
                FROM live_quotes
                ORDER BY ticker, timestamp DESC
            """))
            rows = result.fetchall()

        if rows:
            data = [
                {
                    "ticker":    row.ticker,
                    "price":     float(row.price),
                    "timestamp": str(row.timestamp)
                }
                for row in rows
            ]
        else:
            # Fall back to latest close prices
            with engine.connect() as conn:
                result = conn.execute(text("""
                    SELECT DISTINCT ON (ticker)
                        ticker,
                        close as price,
                        date as timestamp
                    FROM fact_prices
                    ORDER BY ticker, date DESC
                """))
                rows = result.fetchall()

            data = [
                {
                    "ticker":    row.ticker,
                    "price":     float(row.price),
                    "timestamp": str(row.timestamp)
                }
                for row in rows
            ]

        return jsonify({
            "count": len(data),
            "data":  data
        }), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500