# ingestion/alpaca_stream.py
# ─────────────────────────────────────────────────────────────
# Streams real-time price quotes from Alpaca WebSocket
# and writes them to live_quotes table every few seconds
# This runs continuously in the background while markets are open
# ─────────────────────────────────────────────────────────────

import os
import time
import threading
from datetime import datetime
from dotenv import load_dotenv
from sqlalchemy import text
from database.db import engine, test_connection

# Load environment variables
load_dotenv()

# ── WATCHLIST ─────────────────────────────────────────────────
TICKERS = [
    "AAPL", "MSFT", "NVDA", "GOOGL", "AMZN",
    "META", "TSLA", "SPY",  "QQQ",  "JPM",
    "BAC",  "UNH",  "JNJ",  "XOM",  "AMD"
]

# ── BUFFER ────────────────────────────────────────────────────
# Incoming quotes are buffered here and flushed to DB every 5 seconds
# This reduces DB writes from potentially thousands per second to one batch
quote_buffer = {}
buffer_lock  = threading.Lock()


def insert_quotes(quotes: dict):
    # Writes buffered quotes to live_quotes table
    # Each ticker gets one row with its latest price
    if not quotes:
        return

    with engine.connect() as conn:
        for ticker, data in quotes.items():
            try:
                conn.execute(
                    text("""
                        INSERT INTO live_quotes (ticker, price, volume, timestamp)
                        VALUES (:ticker, :price, :volume, :timestamp)
                    """),
                    {
                        "ticker":    ticker,
                        "price":     data.get("price"),
                        "volume":    data.get("volume", 0),
                        "timestamp": data.get("timestamp", datetime.now())
                    }
                )
            except Exception as e:
                print(f"  WARNING: Could not insert quote for {ticker}: {e}")

        conn.commit()


def flush_buffer():
    # Runs every 5 seconds in a background thread
    # Copies the buffer, clears it, then writes to DB
    while True:
        time.sleep(5)
        with buffer_lock:
            if quote_buffer:
                snapshot = quote_buffer.copy()
                quote_buffer.clear()
                insert_quotes(snapshot)
                print(f"  Flushed {len(snapshot)} quotes at {datetime.now().strftime('%H:%M:%S')}")


def get_latest_quotes():
    # Falls back to yfinance for latest prices when market is closed
    # This keeps the dashboard showing real prices even outside market hours
    import yfinance as yf
    print("  Fetching latest prices via yfinance fallback...")

    quotes = {}
    for ticker in TICKERS:
        try:
            data  = yf.Ticker(ticker)
            info  = data.fast_info
            price = info.last_price
            if price:
                quotes[ticker] = {
                    "price":     float(price),
                    "volume":    0,
                    "timestamp": datetime.now()
                }
                print(f"  {ticker}: ${price:.2f}")
        except Exception as e:
            print(f"  WARNING: Could not fetch {ticker}: {e}")

    return quotes


def run_stream():
    # Attempts to connect to Alpaca WebSocket stream
    # Falls back to yfinance snapshot if market is closed
    # or outside trading hours (9:30am - 4pm ET weekdays)
    try:
        from alpaca.data.live import StockDataStream
        from alpaca.data.enums import DataFeed

        api_key    = os.getenv("ALPACA_API_KEY")
        secret_key = os.getenv("ALPACA_SECRET_KEY")

        if not api_key or not secret_key:
            raise ValueError("Alpaca API keys not found in .env")

        print("  Connecting to Alpaca WebSocket...")

        # Initialize the stream
        stream = StockDataStream(api_key, secret_key, feed=DataFeed.IEX)

        async def handle_quote(data):
            # Called for every incoming quote tick
            # Updates the buffer with latest price for that ticker
            with buffer_lock:
                quote_buffer[data.symbol] = {
                    "price":     float(data.ask_price or data.bid_price or 0),
                    "volume":    0,
                    "timestamp": datetime.now()
                }

        # Subscribe to quotes for all tickers
        stream.subscribe_quotes(handle_quote, *TICKERS)

        print(f"  Streaming quotes for {len(TICKERS)} tickers...")
        print("  Press Ctrl+C to stop\n")

        # Start flush thread in background
        flush_thread = threading.Thread(target=flush_buffer, daemon=True)
        flush_thread.start()

        # Start the stream — this blocks until Ctrl+C
        stream.run()

    except Exception as e:
        print(f"  WebSocket unavailable ({e})")
        print("  Falling back to yfinance snapshot mode...")
        run_snapshot_mode()


def run_snapshot_mode():
    # Runs when market is closed or WebSocket is unavailable
    # Polls yfinance every 60 seconds for latest prices
    print("  Snapshot mode — refreshing every 60 seconds")
    print("  Press Ctrl+C to stop\n")

    while True:
        quotes = get_latest_quotes()
        insert_quotes(quotes)
        print(f"  Snapshot written at {datetime.now().strftime('%H:%M:%S')}")
        time.sleep(60)


def run():
    # Main entry point
    print("=" * 55)
    print("KairosAI — Alpaca price stream")
    print(f"Tickers: {len(TICKERS)}")
    print("=" * 55)

    # Verify database connection
    test_connection()

    # Try live stream first, fall back to snapshot
    run_stream()


if __name__ == "__main__":
    run()