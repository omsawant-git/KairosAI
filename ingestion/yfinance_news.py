# ingestion/yfinance_news.py
# ─────────────────────────────────────────────────────────────
# Pulls news headlines from yfinance for all tickers
# Supplements Finnhub data in fact_news
# yfinance news: no API key, no rate limit, free forever
# ─────────────────────────────────────────────────────────────

import time
import yfinance as yf
import pandas as pd
from datetime import datetime
from sqlalchemy import text
from database.db import engine, test_connection


# ── WATCHLIST ─────────────────────────────────────────────────
TICKERS = [
    "AAPL", "MSFT", "NVDA", "GOOGL", "AMZN",
    "META", "TSLA", "SPY",  "QQQ",  "JPM",
    "BAC",  "UNH",  "JNJ",  "XOM",  "AMD"
]


def fetch_news(ticker: str) -> list:
    # Fetches news from yfinance for a single ticker
    # Returns list of article dicts
    try:
        t     = yf.Ticker(ticker)
        news  = t.news
        return news if news else []
    except Exception as e:
        print(f"  WARNING: Could not fetch news for {ticker}: {e}")
        return []


def parse_articles(ticker: str, articles: list) -> list:
    # Parses raw yfinance news into clean dicts
    # yfinance returns nested structure — we flatten it
    parsed = []

    for article in articles:
        try:
            # yfinance news structure varies by version
            # Handle both old and new formats
            content = article.get("content", article)

            # Extract headline
            headline = (
                content.get("title") or
                content.get("headline") or
                article.get("title") or
                ""
            )

            if not headline or len(headline) < 10:
                continue

            # Extract source
            provider = content.get("provider", {})
            if isinstance(provider, dict):
                source = provider.get("displayName", "yfinance")
            else:
                source = str(provider) or "yfinance"

            # Extract URL
            url = (
                content.get("canonicalUrl", {}).get("url") or
                content.get("url") or
                article.get("link") or
                ""
            )
            if isinstance(url, dict):
                url = url.get("url", "")

            # Extract published timestamp
            pub_time = (
                content.get("pubDate") or
                content.get("publishedAt") or
                article.get("providerPublishTime")
            )

            if isinstance(pub_time, (int, float)):
                # Unix timestamp
                published_at = datetime.fromtimestamp(pub_time)
            elif isinstance(pub_time, str):
                # ISO string
                try:
                    published_at = pd.to_datetime(pub_time)
                except Exception:
                    published_at = datetime.now()
            else:
                published_at = datetime.now()

            parsed.append({
                "ticker":       ticker,
                "headline":     str(headline)[:500],
                "source":       str(source)[:255],
                "url":          str(url)[:500],
                "published_at": published_at
            })

        except Exception as e:
            print(f"  WARNING: Could not parse article: {e}")
            continue

    return parsed


def insert_articles(articles: list) -> int:
    # Inserts parsed articles into fact_news
    # Skips duplicates based on headline + ticker
    if not articles:
        return 0

    inserted = 0

    with engine.connect() as conn:
        for article in articles:
            try:
                # Check if headline already exists for this ticker
                existing = conn.execute(
                    text("""
                        SELECT id FROM fact_news
                        WHERE ticker = :ticker
                        AND headline = :headline
                        LIMIT 1
                    """),
                    {
                        "ticker":   article["ticker"],
                        "headline": article["headline"]
                    }
                ).fetchone()

                if existing:
                    continue

                conn.execute(
                    text("""
                        INSERT INTO fact_news
                            (ticker, headline, source, url, published_at)
                        VALUES
                            (:ticker, :headline, :source, :url, :published_at)
                    """),
                    article
                )
                inserted += 1

            except Exception as e:
                print(f"  WARNING: Insert failed: {e}")
                continue

        conn.commit()

    return inserted


def run():
    # Main entry point — fetches and stores news for all tickers
    print("=" * 55)
    print("KairosAI — yfinance news ingestion")
    print(f"Tickers: {len(TICKERS)}")
    print("=" * 55)

    test_connection()

    total_inserted = 0
    total_fetched  = 0

    for ticker in TICKERS:
        print(f"\n  {ticker}")

        # Fetch articles
        articles = fetch_news(ticker)
        print(f"  Fetched {len(articles)} articles")
        total_fetched += len(articles)

        if not articles:
            continue

        # Parse
        parsed = parse_articles(ticker, articles)
        print(f"  Parsed {len(parsed)} valid articles")

        # Insert
        inserted = insert_articles(parsed)
        print(f"  Inserted {inserted} new articles")
        total_inserted += inserted

        # Small delay to be respectful
        time.sleep(0.5)

    print(f"\n{'=' * 55}")
    print(f"Total fetched  : {total_fetched}")
    print(f"Total inserted : {total_inserted}")
    print(f"yfinance news ingestion complete.")
    print("=" * 55)


if __name__ == "__main__":
    run()