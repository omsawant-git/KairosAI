# models/sentiment.py
# ─────────────────────────────────────────────────────────────
# KairosAI Sentiment Analysis — v2
# Model: FinBERT (ProsusAI/finbert)
# Improvements over v1:
#   - Confidence threshold (>65%) — removes noisy predictions
#   - Recency weighting — recent news matters more
#   - Neutral headlines excluded from daily average
#   - Minimum 3 headlines per day filter
#   - Better headline cleaning
#   - Compound score: confidence × direction × recency
#   - Volatility-adjusted sentiment
#   - Sector-level aggregation
#   - 7-day rolling sentiment smoothing
#   - Meaningful evaluation with minimum overlap requirement
# ─────────────────────────────────────────────────────────────

import warnings
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from sqlalchemy import text
from database.db import engine, test_connection
from transformers import pipeline
import torch

warnings.filterwarnings("ignore")


# ── SETTINGS ──────────────────────────────────────────────────
TICKERS = [
    "AAPL", "MSFT", "NVDA", "GOOGL", "AMZN",
    "META", "TSLA", "SPY",  "QQQ",  "JPM",
    "BAC",  "UNH",  "JNJ",  "XOM",  "AMD"
]

# Sector mapping for sector-level aggregation
SECTORS = {
    "Technology":    ["AAPL", "MSFT", "NVDA", "GOOGL", "AMZN", "META", "AMD"],
    "Consumer":      ["TSLA"],
    "Finance":       ["JPM", "BAC"],
    "Healthcare":    ["UNH", "JNJ"],
    "Energy":        ["XOM"],
    "ETF":           ["SPY", "QQQ"]
}

MODEL_NAME         = "ProsusAI/finbert"
BATCH_SIZE         = 32
MAX_LENGTH         = 512
CONFIDENCE_THRESH  = 0.65    # minimum confidence to use prediction
MIN_HEADLINES_DAY  = 3       # minimum headlines per day
RECENCY_DECAY_DAYS = 30      # half-life for recency weighting
ROLLING_WINDOW     = 7       # days for rolling sentiment smoothing
MIN_EVAL_DAYS      = 10      # minimum days needed for evaluation
DEVICE             = 0 if torch.cuda.is_available() else -1

LABEL_SCORES = {
    "positive":  1.0,
    "negative": -1.0,
    "neutral":   0.0
}


# ── LOAD MODEL ────────────────────────────────────────────────

def load_finbert():
    print("  Loading FinBERT...")
    classifier = pipeline(
        "text-classification",
        model=MODEL_NAME,
        tokenizer=MODEL_NAME,
        device=DEVICE,
        truncation=True,
        max_length=MAX_LENGTH
    )
    print(f"  FinBERT loaded ({'GPU' if DEVICE == 0 else 'CPU'})")
    return classifier


# ── DATA LOADING ──────────────────────────────────────────────

def clean_headline(headline: str) -> str:
    # Cleans a headline before scoring
    # Removes URLs, extra whitespace, and garbage text
    import re
    headline = str(headline)
    headline = re.sub(r'http\S+', '', headline)       # remove URLs
    headline = re.sub(r'\s+', ' ', headline).strip()  # normalize whitespace
    headline = re.sub(r'[^\w\s\.,!?-]', '', headline) # remove special chars
    return headline


def load_headlines(ticker: str) -> pd.DataFrame:
    # Loads headlines with better filtering
    with engine.connect() as conn:
        result = conn.execute(
            text("""
                SELECT
                    id,
                    headline,
                    published_at::date AS date,
                    published_at
                FROM fact_news
                WHERE ticker     = :ticker
                AND headline     IS NOT NULL
                AND LENGTH(headline) > 20
                AND headline     NOT LIKE '%http%'
                AND headline     NOT LIKE '%www.%'
                ORDER BY published_at ASC
            """),
            {"ticker": ticker}
        )
        rows = result.fetchall()

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows, columns=["id", "headline", "date", "published_at"])
    df["date"]         = pd.to_datetime(df["date"])
    df["published_at"] = pd.to_datetime(df["published_at"])

    # Clean headlines
    df["headline"] = df["headline"].apply(clean_headline)

    # Remove very short headlines after cleaning
    df = df[df["headline"].str.len() > 15]

    return df


def load_prices(ticker: str) -> pd.DataFrame:
    with engine.connect() as conn:
        result = conn.execute(
            text("""
                SELECT date, close, high, low
                FROM fact_prices
                WHERE ticker = :ticker
                ORDER BY date ASC
            """),
            {"ticker": ticker}
        )
        rows = result.fetchall()

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows, columns=["date", "close", "high", "low"])
    df["date"]  = pd.to_datetime(df["date"])
    df["close"] = df["close"].astype(float)

    # Daily volatility — used for volatility adjustment
    df["volatility"] = (
        (df["high"].astype(float) - df["low"].astype(float)) /
        df["close"]
    ).rolling(10).mean()

    return df


# ── SCORING ───────────────────────────────────────────────────

def score_headlines(classifier, headlines: list) -> list:
    # Runs FinBERT on batches of headlines
    results = []
    for i in range(0, len(headlines), BATCH_SIZE):
        batch = headlines[i:i+BATCH_SIZE]
        try:
            outputs = classifier(batch)
            results.extend(outputs)
        except Exception as e:
            print(f"    WARNING: Batch failed: {e}")
            results.extend([{"label": "neutral", "score": 0.5}] * len(batch))
    return results


def convert_to_compound_score(results: list) -> list:
    # Converts FinBERT output to compound numeric score
    # Compound score = direction × confidence
    # Low confidence predictions → 0.0 (treated as neutral)
    # Neutral labels → 0.0 (excluded from signal)
    scores = []
    for r in results:
        label      = r["label"].lower()
        confidence = r["score"]

        # Skip low confidence — adds noise
        if confidence < CONFIDENCE_THRESH:
            scores.append(0.0)
            continue

        # Neutral headlines excluded from signal
        if label == "neutral":
            scores.append(0.0)
            continue

        direction = LABEL_SCORES.get(label, 0.0)
        scores.append(direction * confidence)

    return scores


# ── AGGREGATION ───────────────────────────────────────────────

def aggregate_daily_sentiment(
    df:     pd.DataFrame,
    scores: list,
    price_df: pd.DataFrame
) -> pd.DataFrame:
    # Aggregates headline scores to daily ticker scores
    # Applies recency weighting and volatility adjustment
    df = df.copy()
    df["raw_score"] = scores
    df["abs_score"] = np.abs(scores)

    # Recency weight — exponential decay from max date
    max_date        = df["date"].max()
    df["days_ago"]  = (max_date - df["date"]).dt.days
    df["recency_w"] = np.exp(-df["days_ago"] / RECENCY_DECAY_DAYS)

    # Combined weight = abs_score × recency
    df["weight"] = df["abs_score"] * df["recency_w"]

    def weighted_sentiment(group):
        # Only use non-neutral scores
        non_neutral = group[group["abs_score"] > 0.01]

        if len(non_neutral) < 1:
            return 0.0

        total_weight = non_neutral["weight"].sum()
        if total_weight < 0.001:
            return 0.0

        return np.average(
            non_neutral["raw_score"],
            weights=non_neutral["weight"]
        )

    daily = df.groupby("date").apply(
        lambda g: pd.Series({
            "sentiment_score": weighted_sentiment(g),
            "n_headlines":     len(g),
            "n_non_neutral":   (g["abs_score"] > 0.01).sum(),
            "pct_positive":    (g["raw_score"] > 0.1).mean(),
            "pct_negative":    (g["raw_score"] < -0.1).mean(),
            "avg_confidence":  g[g["abs_score"] > 0.01]["abs_score"].mean()
            if (g["abs_score"] > 0.01).any() else 0.0
        })
    ).reset_index()

    # Filter days with too few headlines
    daily = daily[daily["n_headlines"] >= MIN_HEADLINES_DAY]

    # Clip to [-1, 1]
    daily["sentiment_score"] = daily["sentiment_score"].clip(-1, 1)

    # ── Volatility adjustment ──
    # High volatility days amplify sentiment signal
    if not price_df.empty:
        daily = daily.merge(
            price_df[["date", "volatility"]],
            on="date", how="left"
        )
        vol_mean = price_df["volatility"].mean()
        vol_std  = price_df["volatility"].std()

        if vol_std > 0:
            daily["vol_adjusted_score"] = daily["sentiment_score"] * (
                1 + (daily["volatility"].fillna(vol_mean) - vol_mean) / (vol_std + 1e-10)
            ).clip(0.5, 2.0)
        else:
            daily["vol_adjusted_score"] = daily["sentiment_score"]
    else:
        daily["vol_adjusted_score"] = daily["sentiment_score"]

    daily["vol_adjusted_score"] = daily["vol_adjusted_score"].clip(-1, 1)

    # ── 7-day rolling sentiment ──
    # Smooths out daily noise
    daily = daily.sort_values("date")
    daily["rolling_sentiment"] = (
        daily["sentiment_score"]
        .rolling(window=ROLLING_WINDOW, min_periods=1)
        .mean()
    )

    return daily


# ── SECTOR AGGREGATION ────────────────────────────────────────

def compute_sector_sentiment(all_daily: dict) -> pd.DataFrame:
    # Aggregates ticker-level sentiment to sector level
    # Gives a broader market signal per sector
    sector_rows = []

    for sector, tickers in SECTORS.items():
        sector_dfs = []
        for ticker in tickers:
            if ticker in all_daily and not all_daily[ticker].empty:
                df = all_daily[ticker][["date", "sentiment_score"]].copy()
                df["ticker"] = ticker
                sector_dfs.append(df)

        if not sector_dfs:
            continue

        combined = pd.concat(sector_dfs)
        daily_sector = combined.groupby("date")["sentiment_score"].mean().reset_index()
        daily_sector["sector"] = sector
        sector_rows.append(daily_sector)

    if not sector_rows:
        return pd.DataFrame()

    return pd.concat(sector_rows)


# ── EVALUATION ────────────────────────────────────────────────

def evaluate_sentiment(
    daily_df: pd.DataFrame,
    price_df: pd.DataFrame
) -> dict:
    # Evaluates if sentiment predicts next-day price direction
    # Requires minimum overlap days for meaningful metrics
    if price_df.empty or len(daily_df) < MIN_EVAL_DAYS:
        return {}

    price_df = price_df.copy()
    price_df["next_day_ret"] = price_df["close"].pct_change().shift(-1)

    merged = daily_df.merge(
        price_df[["date", "next_day_ret"]],
        on="date", how="inner"
    ).dropna(subset=["next_day_ret"])

    if len(merged) < MIN_EVAL_DAYS:
        return {}

    # Only evaluate on non-neutral days
    non_neutral = merged[np.abs(merged["sentiment_score"]) > 0.05]

    if len(non_neutral) < 5:
        return {}

    sent_dir = np.sign(non_neutral["sentiment_score"])
    ret_dir  = np.sign(non_neutral["next_day_ret"])
    dir_acc  = float(np.mean(sent_dir == ret_dir) * 100)

    # Correlation
    corr = float(merged["sentiment_score"].corr(merged["next_day_ret"]))

    # Rolling sentiment evaluation
    if "rolling_sentiment" in merged.columns:
        roll_dir     = np.sign(merged["rolling_sentiment"].dropna())
        ret_dir_roll = np.sign(merged.loc[merged["rolling_sentiment"].notna(), "next_day_ret"])
        n            = min(len(roll_dir), len(ret_dir_roll))
        roll_acc     = float(np.mean(roll_dir.values[:n] == ret_dir_roll.values[:n]) * 100)
    else:
        roll_acc = 0.0

    return {
        "directional_accuracy":         round(dir_acc,  2),
        "rolling_directional_accuracy": round(roll_acc, 2),
        "correlation":                  round(corr,     4),
        "n_days":                       len(merged),
        "n_non_neutral":                len(non_neutral)
    }


# ── SAVE ──────────────────────────────────────────────────────

def save_sentiment(ticker: str, daily_df: pd.DataFrame) -> int:
    saved = 0

    with engine.connect() as conn:
        for _, row in daily_df.iterrows():
            try:
                conn.execute(
                    text("""
                        INSERT INTO fact_signals
                            (ticker, date, sentiment_score)
                        VALUES
                            (:ticker, :date, :sentiment_score)
                        ON CONFLICT (ticker, date)
                        DO UPDATE SET
                            sentiment_score = EXCLUDED.sentiment_score
                    """),
                    {
                        "ticker":          ticker,
                        "date":            row["date"].date(),
                        "sentiment_score": float(row["sentiment_score"])
                    }
                )
                saved += 1
            except Exception as e:
                print(f"    WARNING: {e}")
                continue

        conn.commit()

    return saved


def save_sector_sentiment(sector_df: pd.DataFrame):
    # Saves sector sentiment to a dedicated table
    # Creates table if it doesn't exist
    if sector_df.empty:
        return

    with engine.connect() as conn:
        # Create sector sentiment table if needed
        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS sector_sentiment (
                id          SERIAL PRIMARY KEY,
                sector      VARCHAR(50) NOT NULL,
                date        DATE NOT NULL,
                sentiment   NUMERIC(5,4),
                created_at  TIMESTAMPTZ DEFAULT NOW(),
                UNIQUE(sector, date)
            )
        """))
        conn.commit()

        for _, row in sector_df.iterrows():
            try:
                conn.execute(
                    text("""
                        INSERT INTO sector_sentiment (sector, date, sentiment)
                        VALUES (:sector, :date, :sentiment)
                        ON CONFLICT (sector, date)
                        DO UPDATE SET sentiment = EXCLUDED.sentiment
                    """),
                    {
                        "sector":    row["sector"],
                        "date":      pd.Timestamp(row["date"]).date(),
                        "sentiment": float(row["sentiment_score"])
                    }
                )
            except Exception:
                continue

        conn.commit()


# ── MAIN ──────────────────────────────────────────────────────

def run():
    print("=" * 60)
    print("KairosAI — Sentiment Analysis v2")
    print("Model   : FinBERT (ProsusAI/finbert)")
    print("Fixes   : confidence threshold, recency weighting,")
    print("          neutral exclusion, volatility adjustment,")
    print("          7-day rolling smoothing, sector aggregation")
    print("=" * 60)

    test_connection()
    classifier = load_finbert()

    all_results = []
    all_daily   = {}

    for ticker in TICKERS:
        print(f"\n── {ticker} ────────────────────────────────────")

        # Load data
        df       = load_headlines(ticker)
        price_df = load_prices(ticker)

        if df.empty:
            print(f"  No headlines found, skipping")
            continue

        print(f"  Loaded {len(df)} headlines")

        # Score
        print(f"  Scoring with FinBERT...")
        headlines   = df["headline"].tolist()
        raw_results = score_headlines(classifier, headlines)
        scores      = convert_to_compound_score(raw_results)

        # Label stats
        labels     = [r["label"].lower() for r in raw_results]
        n_pos      = labels.count("positive")
        n_neg      = labels.count("negative")
        n_neu      = labels.count("neutral")
        used       = sum(1 for s in scores if abs(s) > 0.01)

        print(f"  Labels  — pos: {n_pos} | neg: {n_neg} | neu: {n_neu}")
        print(f"  Used    — {used}/{len(scores)} above confidence threshold")

        # Aggregate
        daily_df = aggregate_daily_sentiment(df, scores, price_df)

        if daily_df.empty:
            print(f"  No days met minimum headline threshold")
            continue

        all_daily[ticker] = daily_df

        print(f"  Days    : {len(daily_df)}")
        print(f"  Score range: [{daily_df['sentiment_score'].min():.4f}, "
              f"{daily_df['sentiment_score'].max():.4f}]")
        print(f"  Mean score : {daily_df['sentiment_score'].mean():.4f}")

        # Recent sentiment
        recent = daily_df.tail(5)
        print(f"\n  Recent daily sentiment:")
        for _, row in recent.iterrows():
            sign = "+" if row["sentiment_score"] > 0 else ""
            bar  = "█" * int(abs(row["sentiment_score"]) * 10)
            print(f"    {row['date'].strftime('%Y-%m-%d')}  "
                  f"{sign}{row['sentiment_score']:.4f}  {bar}")

        # Evaluate
        metrics = evaluate_sentiment(daily_df, price_df)
        if metrics:
            print(f"\n  Evaluation:")
            print(f"    Directional accuracy         : {metrics['directional_accuracy']}%")
            print(f"    Rolling directional accuracy : {metrics['rolling_directional_accuracy']}%")
            print(f"    Correlation                  : {metrics['correlation']}")
            print(f"    Non-neutral days             : {metrics['n_non_neutral']}")
            all_results.append({"ticker": ticker, **metrics})

        # Save
        saved = save_sentiment(ticker, daily_df)
        print(f"\n  Saved {saved} daily scores")

    # Sector aggregation
    print(f"\n── Sector sentiment ─────────────────────────────")
    sector_df = compute_sector_sentiment(all_daily)
    if not sector_df.empty:
        save_sector_sentiment(sector_df)
        print(f"  Sector sentiment saved for {sector_df['sector'].nunique()} sectors")
        sector_summary = sector_df.groupby("sector")["sentiment_score"].mean()
        for sector, score in sector_summary.items():
            sign = "+" if score > 0 else ""
            print(f"  {sector:12s} : {sign}{score:.4f}")

    # Summary
    print("\n" + "=" * 60)
    print("SENTIMENT ANALYSIS v2 COMPLETE")
    print("=" * 60)

    if all_results:
        results_df = pd.DataFrame(all_results)
        print("\nSentiment predictive quality:")
        print(results_df[[
            "ticker", "directional_accuracy",
            "rolling_directional_accuracy",
            "correlation", "n_non_neutral"
        ]].to_string(index=False))

        avg_dir      = results_df["directional_accuracy"].mean()
        avg_roll_dir = results_df["rolling_directional_accuracy"].mean()
        avg_corr     = results_df["correlation"].mean()
        above        = (results_df["directional_accuracy"] > 50).sum()

        print(f"\nAverage directional accuracy         : {avg_dir:.2f}%")
        print(f"Average rolling directional accuracy : {avg_roll_dir:.2f}%")
        print(f"Average correlation                  : {avg_corr:.4f}")
        print(f"Tickers above baseline               : {above}/{len(results_df)}")
        print(f"Random baseline                      : 50.00%")
    else:
        print("\nNot enough data for evaluation.")
        print("Sentiment scores saved — evaluation will improve as data accumulates.")

    return all_results


if __name__ == "__main__":
    run()