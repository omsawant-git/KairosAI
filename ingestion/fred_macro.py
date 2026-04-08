# ingestion/fred_macro.py
# ─────────────────────────────────────────────────────────────
# Fetches macroeconomic data from FRED API
# Stores GDP, CPI, interest rates, unemployment in fact_macro
# FRED is the Federal Reserve's free economic data service
# ─────────────────────────────────────────────────────────────

import os
from fredapi import Fred
from datetime import datetime, timedelta
from sqlalchemy import text
from database.db import engine, test_connection
from dotenv import load_dotenv

# Load API key from .env
load_dotenv()

# ── MACRO SERIES ──────────────────────────────────────────────
# Each entry is a FRED series ID and a human readable description
# These are the most important macro indicators for stock analysis
SERIES = {
    "GDP":        "Gross Domestic Product",
    "CPIAUCSL":   "Consumer Price Index (Inflation)",
    "FEDFUNDS":   "Federal Funds Rate (Interest Rate)",
    "UNRATE":     "Unemployment Rate",
    "T10YIE":     "10-Year Breakeven Inflation Rate",
    "DGS10":      "10-Year Treasury Yield",
    "VIXCLS":     "CBOE Volatility Index (VIX)",
    "M2SL":       "M2 Money Supply",
}

# ── DATE RANGE ────────────────────────────────────────────────
# Pull 5 years to match our price history
END_DATE   = datetime.today().strftime("%Y-%m-%d")
START_DATE = (datetime.today() - timedelta(days=365 * 5)).strftime("%Y-%m-%d")


def get_client():
    # Creates and returns a FRED API client
    api_key = os.getenv("FRED_API_KEY")
    if not api_key:
        raise ValueError("FRED_API_KEY not found in .env")
    return Fred(api_key=api_key)


def fetch_series(fred, series_id: str, description: str):
    # Fetches a single macro series from FRED
    # Returns a pandas Series with date index
    try:
        print(f"  Fetching {series_id} — {description}...")
        data = fred.get_series(
            series_id,
            observation_start=START_DATE,
            observation_end=END_DATE
        )
        return data
    except Exception as e:
        print(f"  WARNING: Could not fetch {series_id}: {e}")
        return None


def insert_series(series_id: str, data):
    # Inserts a macro series into fact_macro
    # Skips duplicates using ON CONFLICT DO NOTHING
    if data is None or data.empty:
        return 0

    rows_inserted = 0

    with engine.connect() as conn:
        for date, value in data.items():
            # Skip null values — FRED sometimes has gaps
            if value is None:
                continue

            try:
                import math
                if math.isnan(float(value)):
                    continue
            except (TypeError, ValueError):
                continue

            try:
                conn.execute(
                    text("""
                        INSERT INTO fact_macro (series_id, date, value)
                        VALUES (:series_id, :date, :value)
                        ON CONFLICT (series_id, date) DO NOTHING
                    """),
                    {
                        "series_id": series_id,
                        "date":      date.strftime("%Y-%m-%d"),
                        "value":     float(value)
                    }
                )
                rows_inserted += 1
            except Exception as e:
                print(f"  WARNING: Could not insert row: {e}")
                continue

        conn.commit()

    return rows_inserted


def run():
    # Main entry point — fetches all macro series and stores them
    print("=" * 55)
    print("KairosAI — FRED macro data ingestion")
    print(f"Date range: {START_DATE} to {END_DATE}")
    print(f"Series: {len(SERIES)}")
    print("=" * 55)

    # Verify database connection
    test_connection()

    # Initialize FRED client
    fred = get_client()

    total_rows = 0

    for series_id, description in SERIES.items():
        data = fetch_series(fred, series_id, description)
        rows = insert_series(series_id, data)
        total_rows += rows
        print(f"  Inserted {rows} rows for {series_id}")

    print(f"\nTotal rows inserted: {total_rows}")
    print("\nMacro data ingestion complete.")
    print("=" * 55)


if __name__ == "__main__":
    run()