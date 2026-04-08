# models/evaluate.py
# ─────────────────────────────────────────────────────────────
# KairosAI Model Evaluation Report
# Pulls all model outputs from fact_signals and
# generates a comprehensive evaluation report
# ─────────────────────────────────────────────────────────────

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from sqlalchemy import text
from database.db import engine, test_connection


# ── SETTINGS ──────────────────────────────────────────────────
TICKERS = [
    "AAPL", "MSFT", "NVDA", "GOOGL", "AMZN",
    "META", "TSLA", "SPY",  "QQQ",  "JPM",
    "BAC",  "UNH",  "JNJ",  "XOM",  "AMD"
]


def load_signals(ticker: str) -> pd.DataFrame:
    # Loads all signals for a ticker from fact_signals
    with engine.connect() as conn:
        result = conn.execute(
            text("""
                SELECT
                    date,
                    predicted_close,
                    anomaly_score,
                    is_anomaly,
                    sentiment_score
                FROM fact_signals
                WHERE ticker = :ticker
                ORDER BY date ASC
            """),
            {"ticker": ticker}
        )
        rows = result.fetchall()

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows, columns=[
        "date", "predicted_close", "anomaly_score",
        "is_anomaly", "sentiment_score"
    ])
    df["date"] = pd.to_datetime(df["date"])
    return df


def load_prices(ticker: str) -> pd.DataFrame:
    # Loads actual prices for comparison
    with engine.connect() as conn:
        result = conn.execute(
            text("""
                SELECT date, close
                FROM fact_prices
                WHERE ticker = :ticker
                ORDER BY date ASC
            """),
            {"ticker": ticker}
        )
        rows = result.fetchall()

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows, columns=["date", "close"])
    df["date"]  = pd.to_datetime(df["date"])
    df["close"] = df["close"].astype(float)
    return df


def evaluate_forecasting(signals: pd.DataFrame, prices: pd.DataFrame) -> dict:
    # Evaluates forecast accuracy against actual prices
    # Predictions are for future dates so we match
    # predicted_close on date X with actual close on date X
    if signals.empty or prices.empty:
        return {}

    # Get rows where predicted_close exists
    forecast_df = signals[["date", "predicted_close"]].dropna()

    if forecast_df.empty:
        return {}

    # Match with actual prices on same date
    merged = forecast_df.merge(
        prices[["date", "close"]],
        on="date", how="inner"
    )

    # Also try matching with next available price date
    # since predictions may be for trading days
    if len(merged) < 3:
        # Try merging on nearest date
        forecast_df["date"] = pd.to_datetime(forecast_df["date"])
        prices["date"]      = pd.to_datetime(prices["date"])

        merged = pd.merge_asof(
            forecast_df.sort_values("date"),
            prices[["date", "close"]].sort_values("date"),
            on="date",
            direction="nearest",
            tolerance=pd.Timedelta("5 days")
        ).dropna()

    if len(merged) < 3:
        return {}

    actuals = merged["close"].values
    preds   = merged["predicted_close"].values.astype(float)

    rmse    = float(np.sqrt(np.mean((actuals - preds) ** 2)))
    mae     = float(np.mean(np.abs(actuals - preds)))
    mape    = float(np.mean(np.abs((actuals - preds) / actuals)) * 100)

    if len(actuals) > 1:
        actual_dir = np.sign(np.diff(actuals))
        pred_dir   = np.sign(np.diff(preds))
        dir_acc    = float(np.mean(actual_dir == pred_dir) * 100)
    else:
        dir_acc = 0.0

    return {
        "rmse":            round(rmse,    2),
        "mae":             round(mae,     2),
        "mape":            round(mape,    2),
        "directional_acc": round(dir_acc, 2),
        "n_predictions":   len(merged)
    }


def evaluate_anomaly(signals: pd.DataFrame, prices: pd.DataFrame) -> dict:
    # Evaluates anomaly detection quality
    if signals.empty or prices.empty:
        return {}

    anomaly_df = signals[["date", "is_anomaly", "anomaly_score"]].dropna()

    if anomaly_df.empty:
        return {}

    n_anomalies = int(anomaly_df["is_anomaly"].sum())
    anomaly_pct = round(float(n_anomalies / len(anomaly_df) * 100), 2)
    avg_score   = round(float(anomaly_df["anomaly_score"].mean()), 4)

    # Check if anomalies correspond to large price moves
    prices = prices.copy()
    prices["return"] = prices["close"].pct_change()
    ret_std = prices["return"].std()

    merged = anomaly_df.merge(prices[["date", "return"]], on="date", how="inner")

    if len(merged) < 5:
        return {
            "n_anomalies": n_anomalies,
            "anomaly_pct": anomaly_pct,
            "avg_score":   avg_score
        }

    # Average return magnitude on anomalous vs normal days
    anomalous_returns = merged[merged["is_anomaly"] == True]["return"].abs().mean()
    normal_returns    = merged[merged["is_anomaly"] == False]["return"].abs().mean()

    return {
        "n_anomalies":        n_anomalies,
        "anomaly_pct":        anomaly_pct,
        "avg_score":          avg_score,
        "anomalous_avg_move": round(float(anomalous_returns) * 100, 3),
        "normal_avg_move":    round(float(normal_returns) * 100, 3),
        "signal_lift":        round(float(anomalous_returns / (normal_returns + 1e-10)), 2)
    }


def evaluate_sentiment_signals(signals: pd.DataFrame, prices: pd.DataFrame) -> dict:
    # Evaluates sentiment signal quality
    if signals.empty or prices.empty:
        return {}

    sent_df = signals[["date", "sentiment_score"]].dropna()

    if len(sent_df) < 5:
        return {}

    prices = prices.copy()
    prices["next_ret"] = prices["close"].pct_change().shift(-1)

    merged = sent_df.merge(
        prices[["date", "next_ret"]], on="date", how="inner"
    ).dropna()

    if len(merged) < 5:
        return {}

    corr    = float(merged["sentiment_score"].corr(merged["next_ret"]))
    avg_pos = float(merged[merged["sentiment_score"] > 0.1]["next_ret"].mean())
    avg_neg = float(merged[merged["sentiment_score"] < -0.1]["next_ret"].mean())

    return {
        "n_days":       len(merged),
        "correlation":  round(corr,    4),
        "avg_pos_ret":  round(avg_pos * 100, 3),
        "avg_neg_ret":  round(avg_neg * 100, 3)
    }


def generate_combined_signal(signals: pd.DataFrame) -> pd.DataFrame:
    # Generates a combined signal score from all three models
    # Score range: -1 (very bearish) to +1 (very bullish)
    if signals.empty:
        return pd.DataFrame()

    df = signals.copy()

    # Normalize anomaly score to [-1, 0]
    # High anomaly = uncertainty = slightly bearish
    if "anomaly_score" in df.columns:
        df["anomaly_signal"] = -df["anomaly_score"].fillna(0) * 0.3

    # Sentiment is already [-1, +1]
    if "sentiment_score" in df.columns:
        df["sentiment_signal"] = df["sentiment_score"].fillna(0) * 0.4

    # Forecast direction signal
    if "predicted_close" in df.columns:
        df["forecast_signal"] = df["predicted_close"].pct_change().fillna(0).clip(-0.1, 0.1) * 3

    # Combined score
    signal_cols = [c for c in ["anomaly_signal", "sentiment_signal", "forecast_signal"]
                   if c in df.columns]

    if signal_cols:
        df["combined_signal"] = df[signal_cols].sum(axis=1).clip(-1, 1)

    return df


def run():
    print("=" * 60)
    print("KairosAI — Model Evaluation Report")
    print(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)

    test_connection()

    all_forecast  = []
    all_anomaly   = []
    all_sentiment = []

    for ticker in TICKERS:
        signals = load_signals(ticker)
        prices  = load_prices(ticker)

        if signals.empty:
            continue

        # Evaluate each model
        forecast_eval  = evaluate_forecasting(signals, prices)
        anomaly_eval   = evaluate_anomaly(signals, prices)
        sentiment_eval = evaluate_sentiment_signals(signals, prices)

        if forecast_eval:
            all_forecast.append({"ticker": ticker, **forecast_eval})
        if anomaly_eval:
            all_anomaly.append({"ticker": ticker, **anomaly_eval})
        if sentiment_eval:
            all_sentiment.append({"ticker": ticker, **sentiment_eval})

    # ── Forecasting Report ────────────────────────────────────
    print("\n── Forecasting Model ───────────────────────────────")
    if all_forecast:
        df = pd.DataFrame(all_forecast)
        print(df[[
            "ticker", "rmse", "mae", "mape", "directional_acc", "n_predictions"
        ]].to_string(index=False))
        print(f"\nAvg RMSE    : ${df['rmse'].mean():.2f}")
        print(f"Avg MAPE    : {df['mape'].mean():.2f}%")
        print(f"Avg Dir Acc : {df['directional_acc'].mean():.2f}%")
    else:
        print("  No forecasting data available")

    # ── Anomaly Report ────────────────────────────────────────
    print("\n── Anomaly Detection ───────────────────────────────")
    if all_anomaly:
        df = pd.DataFrame(all_anomaly)
        if "signal_lift" in df.columns:
            print(df[[
                "ticker", "n_anomalies", "anomaly_pct",
                "anomalous_avg_move", "normal_avg_move", "signal_lift"
            ]].to_string(index=False))
            print(f"\nAvg signal lift : {df['signal_lift'].mean():.2f}x")
            print("(signal lift > 1.0 means anomalous days have larger moves)")
        else:
            print(df[["ticker", "n_anomalies", "anomaly_pct"]].to_string(index=False))
    else:
        print("  No anomaly data available")

    # ── Sentiment Report ──────────────────────────────────────
    print("\n── Sentiment Analysis ──────────────────────────────")
    if all_sentiment:
        df = pd.DataFrame(all_sentiment)
        print(df[[
            "ticker", "n_days", "correlation",
            "avg_pos_ret", "avg_neg_ret"
        ]].to_string(index=False))
        print(f"\nAvg correlation : {df['correlation'].mean():.4f}")
        print("(positive correlation = sentiment predicts price direction)")
    else:
        print("  No sentiment data available")

    print("\n" + "=" * 60)
    print("EVALUATION COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    run()