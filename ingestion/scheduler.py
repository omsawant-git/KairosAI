# ingestion/scheduler.py
# ─────────────────────────────────────────────────────────────
# Runs all ingestion scripts automatically on a schedule
# so you never have to run them manually again
#
# Schedule:
#   yfinance batch    → every day at 6:00am
#   Finnhub news      → every day at 6:30am
#   FRED macro        → every Sunday at 7:00am
#   Live quotes       → every 60 seconds (continuous)
# ─────────────────────────────────────────────────────────────

import schedule
import time
import threading
from datetime import datetime

from ingestion.yfinance_batch import run as run_yfinance
from ingestion.finnhub_news   import run as run_news
from ingestion.fred_macro     import run as run_macro
from ingestion.alpaca_stream  import get_latest_quotes, insert_quotes


def log(msg: str):
    # Simple timestamped logger
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {msg}")


def job_yfinance():
    # Runs yfinance batch ingestion
    # Pulls latest OHLCV for all 15 tickers and upserts into fact_prices
    log("Starting yfinance batch ingestion...")
    try:
        run_yfinance()
        log("yfinance batch complete.")
    except Exception as e:
        log(f"ERROR in yfinance batch: {e}")


def job_news():
    # Runs Finnhub news ingestion
    # Fetches latest headlines for all 15 tickers
    log("Starting Finnhub news ingestion...")
    try:
        run_news()
        log("Finnhub news complete.")
    except Exception as e:
        log(f"ERROR in Finnhub news: {e}")


def job_macro():
    # Runs FRED macro ingestion
    # Fetches latest GDP, CPI, interest rates etc.
    log("Starting FRED macro ingestion...")
    try:
        run_macro()
        log("FRED macro complete.")
    except Exception as e:
        log(f"ERROR in FRED macro: {e}")


def job_live_quotes():
    # Fetches latest prices via yfinance snapshot
    # Runs every 60 seconds to keep live_quotes table fresh
    # During market hours the WebSocket handles this instead
    try:
        quotes = get_latest_quotes()
        insert_quotes(quotes)
        log(f"Live quotes updated — {len(quotes)} tickers")
    except Exception as e:
        log(f"ERROR in live quotes: {e}")


def run_threaded(job_fn):
    # Runs a job in a separate thread so it doesn't
    # block the scheduler from running other jobs on time
    thread = threading.Thread(target=job_fn)
    thread.start()


def run():
    # Main entry point — sets up all schedules and starts the loop
    log("=" * 55)
    log("KairosAI Scheduler starting...")
    log("=" * 55)

    # ── DAILY JOBS ────────────────────────────────────────────
    # Price data — runs every day at 6:00am after market close
    schedule.every().day.at("06:00").do(
        run_threaded, job_yfinance
    )

    # News headlines — runs every day at 6:30am
    schedule.every().day.at("06:30").do(
        run_threaded, job_news
    )

    # Macro data — runs every Sunday at 7:00am
    # FRED data updates weekly so daily is overkill
    schedule.every().sunday.at("07:00").do(
        run_threaded, job_macro
    )

    # ── CONTINUOUS JOBS ───────────────────────────────────────
    # Live quotes — runs every 60 seconds
    schedule.every(60).seconds.do(
        run_threaded, job_live_quotes
    )

    log("Schedule configured:")
    log("  yfinance batch  → daily at 06:00")
    log("  Finnhub news    → daily at 06:30")
    log("  FRED macro      → every Sunday at 07:00")
    log("  Live quotes     → every 60 seconds")
    log("=" * 55)
    log("Scheduler running. Press Ctrl+C to stop.")

    # Run live quotes immediately on startup
    # so dashboard has fresh data right away
    run_threaded(job_live_quotes)

    # Main loop — checks every second if a job is due
    while True:
        schedule.run_pending()
        time.sleep(1)


if __name__ == "__main__":
    run()