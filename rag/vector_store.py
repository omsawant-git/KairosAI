# rag/vector_store.py
# ─────────────────────────────────────────────────────────────
# KairosAI RAG Pipeline — Vector Store
# Handles reading and writing vectors to pgvector
# Provides similarity search over document_chunks
# ─────────────────────────────────────────────────────────────

from sqlalchemy import text
from database.db import engine


def search_similar(
    query_embedding: list,
    ticker:          str   = None,
    source:          str   = None,
    top_k:           int   = 5,
    min_similarity:  float = 0.3
) -> list:
    # Searches document_chunks for most similar vectors
    # Uses cosine similarity via pgvector <=> operator
    # Returns top_k most relevant chunks above min_similarity
    filters = ["1=1"]
    params  = {"top_k": top_k}

    if ticker:
        filters.append("ticker = :ticker")
        params["ticker"] = ticker

    if source:
        filters.append("source = :source")
        params["source"] = source

    where = " AND ".join(filters)

    # Convert embedding list to pgvector literal string
    # pgvector requires format: '[0.1, 0.2, ...]'::vector
    embedding_str = "[" + ",".join(str(round(x, 8)) for x in query_embedding) + "]"

    with engine.connect() as conn:
        result = conn.execute(
            text(f"""
                SELECT
                    id,
                    ticker,
                    source,
                    content,
                    1 - (embedding <=> '{embedding_str}'::vector) AS similarity
                FROM document_chunks
                WHERE {where}
                ORDER BY embedding <=> '{embedding_str}'::vector
                LIMIT :top_k
            """),
            params
        )
        rows = result.fetchall()

    if not rows:
        return []

    # Filter by minimum similarity threshold
    results = []
    for row in rows:
        if float(row.similarity) >= min_similarity:
            results.append({
                "id":         row.id,
                "ticker":     row.ticker,
                "source":     row.source,
                "content":    row.content,
                "similarity": round(float(row.similarity), 4)
            })

    return results


def get_chunk_count(ticker: str = None) -> int:
    # Returns total number of chunks in the vector store
    with engine.connect() as conn:
        if ticker:
            result = conn.execute(
                text("SELECT COUNT(*) FROM document_chunks WHERE ticker = :ticker"),
                {"ticker": ticker}
            )
        else:
            result = conn.execute(
                text("SELECT COUNT(*) FROM document_chunks")
            )
        return result.scalar()


def get_recent_chunks(ticker: str, limit: int = 10) -> list:
    # Returns most recently added chunks for a ticker
    with engine.connect() as conn:
        result = conn.execute(
            text("""
                SELECT id, ticker, source, content, created_at
                FROM document_chunks
                WHERE ticker = :ticker
                ORDER BY created_at DESC
                LIMIT :limit
            """),
            {"ticker": ticker, "limit": limit}
        )
        rows = result.fetchall()

    return [
        {
            "id":         row.id,
            "ticker":     row.ticker,
            "source":     row.source,
            "content":    row.content,
            "created_at": str(row.created_at)
        }
        for row in rows
    ]


def delete_ticker_chunks(ticker: str) -> int:
    # Deletes all chunks for a ticker
    # Used when re-embedding after data update
    with engine.connect() as conn:
        result = conn.execute(
            text("DELETE FROM document_chunks WHERE ticker = :ticker"),
            {"ticker": ticker}
        )
        conn.commit()
        return result.rowcount