# ingestion/finnhub_news.py
# ─────────────────────────────────────────────────────────────
# Fetches news headlines from Finnhub for all tickers
# and stores them in fact_news table
# Finnhub free tier: 60 calls/minute — we stay well under that
# ─────────────────────────────────────────────────────────────

import os
import time
import finnhub
from datetime import datetime, timedelta
from sqlalchemy import text
from database.db import engine, test_connection
from dotenv import load_dotenv

# Load API key from .env
load_dotenv()

# ── WATCHLIST ─────────────────────────────────────────────────
TICKERS = [
    "AAPL", "MSFT", "NVDA", "GOOGL", "AMZN",
    "META", "TSLA", "SPY",  "QQQ",  "JPM",
    "BAC",  "UNH",  "JNJ",  "XOM",  "AMD"
]

# ── DATE RANGE ────────────────────────────────────────────────
# Finnhub free tier allows up to 1 year of news history
END_DATE   = datetime.today()
START_DATE = datetime.today() - timedelta(days=365)

# Convert to unix timestamps — Finnhub requires this format
END_UNIX   = int(END_DATE.timestamp())
START_UNIX = int(START_DATE.timestamp())


def get_client():
    # Creates and returns a Finnhub API client
    api_key = os.getenv("FINNHUB_API_KEY")
    if not api_key:
        raise ValueError("FINNHUB_API_KEY not found in .env")
    return finnhub.Client(api_key=api_key)


def fetch_news(client, ticker: str) -> list:
    # Fetches news headlines for a single ticker
    # Returns a list of article dicts from Finnhub
    try:
        news = client.company_news(
            ticker,
            _from=START_DATE.strftime("%Y-%m-%d"),
            to=END_DATE.strftime("%Y-%m-%d")
        )
        return news if news else []
    except Exception as e:
        print(f"  WARNING: Could not fetch news for {ticker}: {e}")
        return []


def insert_news(ticker: str, articles: list) -> int:
    # Inserts news articles into fact_news table
    # Skips duplicates based on headline + ticker + published_at
    if not articles:
        return 0

    rows_inserted = 0

    with engine.connect() as conn:
        for article in articles:
            try:
                # Convert unix timestamp to datetime
                published_at = datetime.fromtimestamp(
                    article.get("datetime", 0)
                )

                conn.execute(
                    text("""
                        INSERT INTO fact_news
                            (ticker, headline, source, url, published_at)
                        VALUES
                            (:ticker, :headline, :source, :url, :published_at)
                        ON CONFLICT DO NOTHING
                    """),
                    {
                        "ticker":       ticker,
                        "headline":     article.get("headline", "")[:500],
                        "source":       article.get("source",   "")[:255],
                        "url":          article.get("url",      "")[:500],
                        "published_at": published_at
                    }
                )
                rows_inserted += 1

            except Exception as e:
                print(f"  WARNING: Could not insert article: {e}")
                continue

        conn.commit()

    return rows_inserted


def run():
    # Main entry point — fetches and stores news for all tickers
    print("=" * 55)
    print("KairosAI — Finnhub news ingestion")
    print(f"Date range: {START_DATE.strftime('%Y-%m-%d')} to {END_DATE.strftime('%Y-%m-%d')}")
    print(f"Tickers: {len(TICKERS)}")
    print("=" * 55)

    # Verify database connection
    test_connection()

    # Initialize Finnhub client
    client = get_client()

    total_articles = 0

    for ticker in TICKERS:
        print(f"  Fetching news for {ticker}...")

        articles = fetch_news(client, ticker)
        rows     = insert_news(ticker, articles)
        total_articles += rows

        print(f"  Inserted {rows} articles for {ticker}")

        # Sleep 1 second between tickers to respect rate limits
        # Finnhub free tier = 60 calls/minute, we use 1 call per ticker
        time.sleep(1)

    print(f"\nTotal articles inserted: {total_articles}")
    print("\nNews ingestion complete.")
    print("=" * 55)


if __name__ == "__main__":
    run()