# ingestion/yfinance_batch.py
# ─────────────────────────────────────────────────────────────
# Pulls 2 years of historical OHLCV data from Yahoo Finance
# for all tickers in the watchlist and loads into fact_prices
# ─────────────────────────────────────────────────────────────

import yfinance as yf
import pandas as pd
from sqlalchemy import text
from database.db import engine, test_connection
from datetime import datetime, timedelta

# ── WATCHLIST ─────────────────────────────────────────────────
# 15 tickers covering tech, finance, healthcare, energy, benchmarks
TICKERS = [
    "AAPL",   # Apple
    "MSFT",   # Microsoft
    "NVDA",   # NVIDIA
    "GOOGL",  # Alphabet
    "AMZN",   # Amazon
    "META",   # Meta
    "TSLA",   # Tesla
    "SPY",    # S&P 500 ETF
    "QQQ",    # Nasdaq 100 ETF
    "JPM",    # JPMorgan Chase
    "BAC",    # Bank of America
    "UNH",    # UnitedHealth
    "JNJ",    # Johnson & Johnson
    "XOM",    # ExxonMobil
    "AMD",    # AMD
]

# ── DATE RANGE ────────────────────────────────────────────────
# Pull 2 years of historical data
END_DATE   = datetime.today().strftime("%Y-%m-%d")
START_DATE = (datetime.today() - timedelta(days=365 * 5)).strftime("%Y-%m-%d")

def fetch_ticker(ticker: str) -> pd.DataFrame:
    # Downloads OHLCV data for a single ticker from Yahoo Finance
    # Returns a cleaned DataFrame ready for database insertion
    print(f"  Fetching {ticker}...")

    df = yf.download(
        ticker,
        start=START_DATE,
        end=END_DATE,
        auto_adjust=True,   # adjusts for splits and dividends
        progress=False      # suppresses the yfinance progress bar
    )

    if df.empty:
        print(f"  WARNING: No data returned for {ticker}")
        return pd.DataFrame()

    # Flatten MultiIndex columns if present
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    # Rename columns to match our schema
    df = df.rename(columns={
        "Open":   "open",
        "High":   "high",
        "Low":    "low",
        "Close":  "close",
        "Volume": "volume"
    })

    # Keep only the columns we need
    df = df[["open", "high", "low", "close", "volume"]].copy()

    # Reset index so date becomes a regular column
    df = df.reset_index()
    df = df.rename(columns={"Date": "date"})

    # Add ticker column
    df["ticker"] = ticker

    # Drop any rows with missing close price
    df = df.dropna(subset=["close"])

    return df


def insert_prices(df: pd.DataFrame):
    # Inserts a DataFrame of OHLCV rows into fact_prices
    # Uses ON CONFLICT DO NOTHING to skip duplicates safely
    if df.empty:
        return 0

    rows_inserted = 0

    with engine.connect() as conn:
        for _, row in df.iterrows():
            conn.execute(
                text("""
                    INSERT INTO fact_prices
                        (ticker, date, open, high, low, close, volume)
                    VALUES
                        (:ticker, :date, :open, :high, :low, :close, :volume)
                    ON CONFLICT (ticker, date) DO NOTHING
                """),
                {
                    "ticker": row["ticker"],
                    "date":   row["date"].date() if hasattr(row["date"], "date") else row["date"],
                    "open":   float(row["open"])   if pd.notna(row["open"])   else None,
                    "high":   float(row["high"])   if pd.notna(row["high"])   else None,
                    "low":    float(row["low"])    if pd.notna(row["low"])    else None,
                    "close":  float(row["close"])  if pd.notna(row["close"])  else None,
                    "volume": int(row["volume"])   if pd.notna(row["volume"]) else None,
                }
            )
            rows_inserted += 1

        conn.commit()

    return rows_inserted


def populate_dim_ticker():
    # Inserts ticker metadata (company name, sector, industry)
    # into dim_ticker using yfinance .info
    print("\nPopulating dim_ticker metadata...")

    with engine.connect() as conn:
        for ticker in TICKERS:
            try:
                info = yf.Ticker(ticker).info
                conn.execute(
                    text("""
                        INSERT INTO dim_ticker (ticker, company, sector, industry)
                        VALUES (:ticker, :company, :sector, :industry)
                        ON CONFLICT (ticker) DO NOTHING
                    """),
                    {
                        "ticker":   ticker,
                        "company":  info.get("longName", ""),
                        "sector":   info.get("sector", ""),
                        "industry": info.get("industry", "")
                    }
                )
                print(f"  {ticker} — {info.get('longName', 'N/A')}")
            except Exception as e:
                print(f"  WARNING: Could not fetch info for {ticker}: {e}")

        conn.commit()


def run():
    # Main entry point — runs the full batch ingestion
    print("=" * 55)
    print("KairosAI — yfinance batch ingestion")
    print(f"Date range: {START_DATE} to {END_DATE}")
    print(f"Tickers: {len(TICKERS)}")
    print("=" * 55)

    # Verify database is reachable before starting
    test_connection()

    total_rows = 0

    for ticker in TICKERS:
        df = fetch_ticker(ticker)
        rows = insert_prices(df)
        total_rows += rows
        print(f"  Inserted {rows} rows for {ticker}")

    print(f"\nTotal rows inserted: {total_rows}")

    # Populate ticker metadata after prices
    populate_dim_ticker()

    print("\nBatch ingestion complete.")
    print("=" * 55)


if __name__ == "__main__":
    run()