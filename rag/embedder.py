# rag/embedder.py
# ─────────────────────────────────────────────────────────────
# KairosAI RAG Pipeline — Document Embedder
# Chunks news headlines and SEC-style text into segments
# Embeds using sentence-transformers all-MiniLM-L6-v2 (free, local)
# Stores vectors + metadata into document_chunks table (pgvector)
# ─────────────────────────────────────────────────────────────

import os
import numpy as np
import pandas as pd
from datetime import datetime
from sqlalchemy import text
from database.db import engine, test_connection
from sentence_transformers import SentenceTransformer


# ── SETTINGS ──────────────────────────────────────────────────
TICKERS    = [
    "AAPL", "MSFT", "NVDA", "GOOGL", "AMZN",
    "META", "TSLA", "SPY",  "QQQ",  "JPM",
    "BAC",  "UNH",  "JNJ",  "XOM",  "AMD"
]

# all-MiniLM-L6-v2 produces 384-dimension vectors
# Fast, free, good quality for financial text retrieval
MODEL_NAME  = "all-MiniLM-L6-v2"
BATCH_SIZE  = 64     # headlines per embedding batch
CHUNK_SIZE  = 3      # combine N headlines into one chunk
# Combining headlines into chunks gives more context per vector
# "AAPL beats earnings. iPhone sales up 12%. Guidance raised."
# is a much richer chunk than scoring each headline alone


# ── LOAD MODEL ────────────────────────────────────────────────

def load_embedder():
    # Loads sentence-transformers model locally
    # Downloads ~90MB on first run, cached after that
    print("  Loading sentence-transformers (all-MiniLM-L6-v2)...")
    model = SentenceTransformer(MODEL_NAME)
    print(f"  Model loaded — embedding dimension: {model.get_sentence_embedding_dimension()}")
    return model


# ── DATA LOADING ──────────────────────────────────────────────

def load_news_for_embedding(ticker: str) -> pd.DataFrame:
    # Loads news headlines that haven't been embedded yet
    # Checks document_chunks to avoid re-embedding
    with engine.connect() as conn:
        result = conn.execute(
            text("""
                SELECT
                    fn.id,
                    fn.headline,
                    fn.source,
                    fn.published_at::date AS date
                FROM fact_news fn
                WHERE fn.ticker = :ticker
                AND fn.headline IS NOT NULL
                AND LENGTH(fn.headline) > 20
                AND fn.headline NOT LIKE '%http%'
                ORDER BY fn.published_at ASC
            """),
            {"ticker": ticker}
        )
        rows = result.fetchall()

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows, columns=["id", "headline", "source", "date"])
    df["date"] = pd.to_datetime(df["date"])
    return df


def load_signals_for_embedding(ticker: str) -> pd.DataFrame:
    # Loads ML signal summaries as text for embedding
    # This allows the agent to retrieve signal context
    with engine.connect() as conn:
        result = conn.execute(
            text("""
                SELECT
                    fs.date,
                    fs.predicted_close,
                    fs.anomaly_score,
                    fs.is_anomaly,
                    fs.sentiment_score,
                    fp.close as actual_close
                FROM fact_signals fs
                LEFT JOIN fact_prices fp
                    ON fs.ticker = fp.ticker
                    AND fs.date  = fp.date
                WHERE fs.ticker = :ticker
                AND (
                    fs.is_anomaly = true
                    OR fs.sentiment_score IS NOT NULL
                )
                ORDER BY fs.date DESC
                LIMIT 100
            """),
            {"ticker": ticker}
        )
        rows = result.fetchall()

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows, columns=[
        "date", "predicted_close", "anomaly_score",
        "is_anomaly", "sentiment_score", "actual_close"
    ])
    df["date"] = pd.to_datetime(df["date"])
    return df


# ── CHUNKING ──────────────────────────────────────────────────

def chunk_headlines(df: pd.DataFrame, ticker: str) -> list:
    # Groups headlines into chunks of CHUNK_SIZE
    # Each chunk becomes one vector in the database
    # Returns list of dicts with content and metadata
    chunks  = []
    headlines = df["headline"].tolist()
    dates     = df["date"].tolist()

    for i in range(0, len(headlines), CHUNK_SIZE):
        batch_headlines = headlines[i:i+CHUNK_SIZE]
        batch_dates     = dates[i:i+CHUNK_SIZE]

        # Combine headlines into one text chunk
        content = " | ".join(batch_headlines)

        # Use the most recent date in the chunk
        chunk_date = max(batch_dates) if batch_dates else datetime.now()

        chunks.append({
            "ticker":  ticker,
            "source":  "news",
            "content": content,
            "date":    chunk_date
        })

    return chunks


def signals_to_text(df: pd.DataFrame, ticker: str) -> list:
    # Converts ML signal rows to human-readable text chunks
    # These become searchable documents in the RAG pipeline
    chunks = []

    for _, row in df.iterrows():
        parts = [f"{ticker} on {row['date'].strftime('%Y-%m-%d')}:"]

        if row["actual_close"]:
            parts.append(f"price was ${float(row['actual_close']):.2f}")

        if row["is_anomaly"]:
            score = float(row["anomaly_score"]) if row["anomaly_score"] else 0
            parts.append(f"anomaly detected (score={score:.3f})")

        if row["sentiment_score"] is not None:
            score = float(row["sentiment_score"])
            label = "positive" if score > 0.1 else "negative" if score < -0.1 else "neutral"
            parts.append(f"sentiment was {label} ({score:.3f})")

        if row["predicted_close"]:
            parts.append(f"forecast was ${float(row['predicted_close']):.2f}")

        content = " ".join(parts)

        chunks.append({
            "ticker":  ticker,
            "source":  "signals",
            "content": content,
            "date":    row["date"]
        })

    return chunks


# ── EMBEDDING ─────────────────────────────────────────────────

def embed_chunks(model, chunks: list) -> list:
    # Generates embeddings for a list of chunks
    # Returns chunks with embedding vectors added
    if not chunks:
        return []

    contents = [c["content"] for c in chunks]

    # Batch embed for efficiency
    embeddings = model.encode(
        contents,
        batch_size=BATCH_SIZE,
        show_progress_bar=False,
        normalize_embeddings=True  # L2 normalize for cosine similarity
    )

    for i, chunk in enumerate(chunks):
        chunk["embedding"] = embeddings[i].tolist()

    return chunks


# ── STORAGE ───────────────────────────────────────────────────

def store_chunks(chunks: list) -> int:
    # Stores embedded chunks into document_chunks table
    # Skips duplicates based on content hash
    if not chunks:
        return 0

    stored = 0

    with engine.connect() as conn:
        for chunk in chunks:
            try:
                # Check if this content already exists
                existing = conn.execute(
                    text("""
                        SELECT id FROM document_chunks
                        WHERE ticker  = :ticker
                        AND   source  = :source
                        AND   content = :content
                        LIMIT 1
                    """),
                    {
                        "ticker":  chunk["ticker"],
                        "source":  chunk["source"],
                        "content": chunk["content"][:500]
                    }
                ).fetchone()

                if existing:
                    continue

                # Store with vector embedding
                conn.execute(
                    text("""
                        INSERT INTO document_chunks
                            (ticker, source, content, embedding)
                        VALUES
                            (:ticker, :source, :content, :embedding)
                    """),
                    {
                        "ticker":    chunk["ticker"],
                        "source":    chunk["source"],
                        "content":   chunk["content"][:2000],
                        "embedding": str(chunk["embedding"])
                    }
                )
                stored += 1

            except Exception as e:
                print(f"    WARNING: Could not store chunk: {e}")
                continue

        conn.commit()

    return stored


# ── MAIN ──────────────────────────────────────────────────────

def run():
    print("=" * 60)
    print("KairosAI — Document Embedder")
    print(f"Model  : {MODEL_NAME} (384 dimensions)")
    print(f"Sources: news headlines + ML signals")
    print("=" * 60)

    test_connection()
    model = load_embedder()

    total_stored = 0

    for ticker in TICKERS:
        print(f"\n── {ticker} ────────────────────────────────────")

        # ── Embed news headlines ──
        news_df = load_news_for_embedding(ticker)

        if not news_df.empty:
            print(f"  News: {len(news_df)} headlines")
            news_chunks     = chunk_headlines(news_df, ticker)
            news_embedded   = embed_chunks(model, news_chunks)
            news_stored     = store_chunks(news_embedded)
            print(f"  Stored {news_stored} news chunks")
            total_stored   += news_stored
        else:
            print(f"  No news headlines found")

        # ── Embed ML signals ──
        signals_df = load_signals_for_embedding(ticker)

        if not signals_df.empty:
            print(f"  Signals: {len(signals_df)} rows")
            signal_chunks   = signals_to_text(signals_df, ticker)
            signal_embedded = embed_chunks(model, signal_chunks)
            signal_stored   = store_chunks(signal_embedded)
            print(f"  Stored {signal_stored} signal chunks")
            total_stored   += signal_stored
        else:
            print(f"  No signal data found")

    print(f"\n{'=' * 60}")
    print(f"Total chunks stored: {total_stored}")
    print("Embedding complete.")
    print("=" * 60)


if __name__ == "__main__":
    run()