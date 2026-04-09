# api/routes_predictions.py
# ─────────────────────────────────────────────────────────────
# KairosAI Flask API — Predictions & Signals Routes
# Serves ML model outputs from fact_signals
# Endpoints:
#   GET /signals/<ticker>          — all signals for a ticker
#   GET /signals/<ticker>/latest   — latest complete signal row
#   GET /signals/<ticker>/forecast — price forecast history
#   GET /signals/<ticker>/anomalies — recent anomalies
#   GET /signals/<ticker>/sentiment — sentiment scores
#   GET /signals/summary/all       — all tickers summary
# ─────────────────────────────────────────────────────────────

from flask import Blueprint, jsonify, request
from flask_jwt_extended import jwt_required
from sqlalchemy import text
from database.db import engine

predictions_bp = Blueprint("predictions", __name__, url_prefix="/signals")


def query_db(sql: str, params: dict = {}) -> list:
    # Helper — runs SQL and returns list of dicts
    with engine.connect() as conn:
        result = conn.execute(text(sql), params)
        cols   = result.keys()
        rows   = result.fetchall()
    return [dict(zip(cols, row)) for row in rows]


def serialize_row(row: dict) -> dict:
    # Converts non-serializable types to strings
    for key, val in row.items():
        if hasattr(val, "isoformat"):
            row[key] = str(val)
        elif val is not None:
            try:
                row[key] = float(val) if isinstance(val, (int, float)) else val
            except Exception:
                row[key] = str(val)
    return row


# ── ALL SIGNALS ───────────────────────────────────────────────
@predictions_bp.route("/<ticker>", methods=["GET"])
@jwt_required()
def get_signals(ticker: str):
    # Returns signal rows for a ticker
    # ?days=30 limits to last N days
    days   = request.args.get("days", 30, type=int)
    ticker = ticker.upper()

    rows = query_db("""
        SELECT
            date,
            predicted_close,
            anomaly_score,
            is_anomaly,
            sentiment_score
        FROM fact_signals
        WHERE ticker = :ticker
        AND date >= CURRENT_DATE - :days * INTERVAL '1 day'
        ORDER BY date DESC
    """, {"ticker": ticker, "days": days})

    rows = [serialize_row(r) for r in rows]

    return jsonify({
        "ticker": ticker,
        "days":   days,
        "count":  len(rows),
        "data":   rows
    }), 200


# ── LATEST COMPLETE SIGNAL ────────────────────────────────────
@predictions_bp.route("/<ticker>/latest", methods=["GET"])
@jwt_required()
def get_latest_signal(ticker: str):
    # Returns most recent signal row up to today
    # Filters to CURRENT_DATE so we get historical data
    # which has anomaly + sentiment populated
    ticker = ticker.upper()

    rows = query_db("""
        SELECT
            fs.date,
            fs.predicted_close,
            fs.anomaly_score,
            fs.is_anomaly,
            fs.sentiment_score,
            fp.close  AS current_price,
            dt.company,
            dt.sector
        FROM fact_signals fs
        LEFT JOIN fact_prices fp
            ON fs.ticker = fp.ticker
            AND fp.date = (
                SELECT MAX(date) FROM fact_prices
                WHERE ticker = :ticker
            )
        LEFT JOIN dim_ticker dt
            ON fs.ticker = dt.ticker
        WHERE fs.ticker = :ticker
        AND fs.date <= CURRENT_DATE
        ORDER BY fs.date DESC
        LIMIT 1
    """, {"ticker": ticker})

    if not rows:
        return jsonify({"error": f"No signals found for {ticker}"}), 404

    row           = serialize_row(rows[0])
    row["ticker"] = ticker

    # Add forecast direction from future prediction
    forecast_rows = query_db("""
        SELECT predicted_close, date
        FROM fact_signals
        WHERE ticker = :ticker
        AND date > CURRENT_DATE
        ORDER BY date ASC
        LIMIT 1
    """, {"ticker": ticker})

    if forecast_rows and row.get("current_price"):
        pred    = float(forecast_rows[0]["predicted_close"])
        current = float(row["current_price"])
        row["forecast_price"]  = round(pred, 4)
        row["direction"]       = "UP" if pred > current else "DOWN"
        row["change_pct"]      = round((pred - current) / current * 100, 2)
        row["forecast_date"]   = str(forecast_rows[0]["date"])

    return jsonify(row), 200


# ── FORECAST HISTORY ──────────────────────────────────────────
@predictions_bp.route("/<ticker>/forecast", methods=["GET"])
@jwt_required()
def get_forecast(ticker: str):
    # Returns forecast vs actual price history
    ticker = ticker.upper()

    rows = query_db("""
        SELECT
            fs.date,
            fs.predicted_close,
            fp.close AS actual_close
        FROM fact_signals fs
        LEFT JOIN fact_prices fp
            ON fs.ticker = fp.ticker
            AND fs.date  = fp.date
        WHERE fs.ticker = :ticker
        AND fs.predicted_close IS NOT NULL
        ORDER BY fs.date DESC
        LIMIT 60
    """, {"ticker": ticker})

    rows = [serialize_row(r) for r in rows]

    return jsonify({
        "ticker": ticker,
        "count":  len(rows),
        "data":   rows
    }), 200


# ── ANOMALIES ─────────────────────────────────────────────────
@predictions_bp.route("/<ticker>/anomalies", methods=["GET"])
@jwt_required()
def get_anomalies(ticker: str):
    # Returns anomalous days for a ticker
    days   = request.args.get("days", 60, type=int)
    ticker = ticker.upper()

    rows = query_db("""
        SELECT
            fs.date,
            fs.anomaly_score,
            fs.is_anomaly,
            fp.close,
            fp.volume
        FROM fact_signals fs
        LEFT JOIN fact_prices fp
            ON fs.ticker = fp.ticker
            AND fs.date  = fp.date
        WHERE fs.ticker   = :ticker
        AND fs.is_anomaly = true
        AND fs.date >= CURRENT_DATE - :days * INTERVAL '1 day'
        ORDER BY fs.anomaly_score DESC
        LIMIT 20
    """, {"ticker": ticker, "days": days})

    rows = [serialize_row(r) for r in rows]

    return jsonify({
        "ticker":      ticker,
        "n_anomalies": len(rows),
        "data":        rows
    }), 200


# ── SENTIMENT ─────────────────────────────────────────────────
@predictions_bp.route("/<ticker>/sentiment", methods=["GET"])
@jwt_required()
def get_sentiment(ticker: str):
    # Returns sentiment score history
    days   = request.args.get("days", 30, type=int)
    ticker = ticker.upper()

    rows = query_db("""
        SELECT date, sentiment_score
        FROM fact_signals
        WHERE ticker = :ticker
        AND sentiment_score IS NOT NULL
        AND date >= CURRENT_DATE - :days * INTERVAL '1 day'
        ORDER BY date DESC
    """, {"ticker": ticker, "days": days})

    rows = [serialize_row(r) for r in rows]

    if rows:
        avg_score = sum(
            float(r["sentiment_score"]) for r in rows
        ) / len(rows)
        label = (
            "positive" if avg_score > 0.1
            else "negative" if avg_score < -0.1
            else "neutral"
        )
    else:
        avg_score = 0.0
        label     = "neutral"

    return jsonify({
        "ticker":        ticker,
        "avg_sentiment": round(avg_score, 4),
        "label":         label,
        "data":          rows
    }), 200


# ── SUMMARY ALL TICKERS ───────────────────────────────────────
@predictions_bp.route("/summary/all", methods=["GET"])
@jwt_required()
def get_summary():
    # Returns latest complete signal for all 15 tickers
    rows = query_db("""
        SELECT DISTINCT ON (fs.ticker)
            fs.ticker,
            fs.date,
            fs.predicted_close,
            fs.anomaly_score,
            fs.is_anomaly,
            fs.sentiment_score,
            fp.close  AS current_price,
            dt.company,
            dt.sector
        FROM fact_signals fs
        LEFT JOIN fact_prices fp
            ON fs.ticker = fp.ticker
            AND fp.date = (
                SELECT MAX(date) FROM fact_prices p2
                WHERE p2.ticker = fs.ticker
            )
        LEFT JOIN dim_ticker dt
            ON fs.ticker = dt.ticker
        WHERE fs.date <= CURRENT_DATE
        ORDER BY fs.ticker, fs.date DESC
    """)

    results = []
    for row in rows:
        row = serialize_row(row)
        if row.get("predicted_close") and row.get("current_price"):
            pred    = float(row["predicted_close"])
            current = float(row["current_price"])
            row["direction"]  = "UP" if pred > current else "DOWN"
            row["change_pct"] = round((pred - current) / current * 100, 2)
        results.append(row)

    return jsonify({
        "count": len(results),
        "data":  results
    }), 200