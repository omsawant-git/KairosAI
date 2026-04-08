# rag/retriever.py
# ─────────────────────────────────────────────────────────────
# KairosAI RAG Pipeline — Retriever
# Given a natural language query, finds the most relevant
# document chunks from the vector store
# Used by the AI agent to ground its answers in real data
# ─────────────────────────────────────────────────────────────

import re
from sentence_transformers import SentenceTransformer
from rag.vector_store import search_similar

# ── SETTINGS ──────────────────────────────────────────────────
MODEL_NAME     = "all-MiniLM-L6-v2"
DEFAULT_TOP_K  = 5
MIN_SIMILARITY = 0.25

# Known tickers for query parsing
KNOWN_TICKERS = [
    "AAPL", "MSFT", "NVDA", "GOOGL", "AMZN",
    "META", "TSLA", "SPY",  "QQQ",  "JPM",
    "BAC",  "UNH",  "JNJ",  "XOM",  "AMD"
]

# Company name → ticker mapping for natural language queries
COMPANY_MAP = {
    "apple":      "AAPL",
    "microsoft":  "MSFT",
    "nvidia":     "NVDA",
    "google":     "GOOGL",
    "alphabet":   "GOOGL",
    "amazon":     "AMZN",
    "meta":       "META",
    "facebook":   "META",
    "tesla":      "TSLA",
    "jpmorgan":   "JPM",
    "jp morgan":  "JPM",
    "bank of america": "BAC",
    "unitedhealth": "UNH",
    "johnson":    "JNJ",
    "exxon":      "XOM",
    "amd":        "AMD"
}

# Load model once at module level — reused across all queries
_model = None


def get_model() -> SentenceTransformer:
    # Lazy loads the embedding model
    # Only loads once and caches for reuse
    global _model
    if _model is None:
        _model = SentenceTransformer(MODEL_NAME)
    return _model


def extract_ticker(query: str) -> str:
    # Extracts ticker symbol from a natural language query
    # Checks for explicit tickers first then company names
    query_upper = query.upper()

    # Check for explicit ticker mentions
    for ticker in KNOWN_TICKERS:
        if ticker in query_upper.split():
            return ticker

        # Also check with common punctuation around it
        if f"({ticker})" in query_upper or f"${ticker}" in query_upper:
            return ticker

    # Check for company names
    query_lower = query.lower()
    for company, ticker in COMPANY_MAP.items():
        if company in query_lower:
            return ticker

    return None


def retrieve(
    query:          str,
    ticker:         str   = None,
    top_k:          int   = DEFAULT_TOP_K,
    min_similarity: float = MIN_SIMILARITY,
    source:         str   = None
) -> list:
    # Main retrieval function
    # Embeds the query and finds similar chunks
    # Auto-detects ticker from query if not provided

    # Auto-detect ticker if not specified
    if ticker is None:
        ticker = extract_ticker(query)

    # Embed the query
    model           = get_model()
    query_embedding = model.encode(
        query,
        normalize_embeddings=True
    ).tolist()

    # Search vector store
    results = search_similar(
        query_embedding = query_embedding,
        ticker          = ticker,
        source          = source,
        top_k           = top_k,
        min_similarity  = min_similarity
    )

    return results


def retrieve_for_agent(query: str, ticker: str = None) -> str:
    # Retrieves relevant context and formats it
    # as a clean string for the AI agent prompt
    results = retrieve(query, ticker=ticker, top_k=5)

    if not results:
        return "No relevant context found in the knowledge base."

    # Format results as readable context
    context_parts = []
    for i, r in enumerate(results, 1):
        source_label = "News" if r["source"] == "news" else "Signal"
        ticker_label = r["ticker"] or "Market"
        similarity   = r["similarity"]

        context_parts.append(
            f"[{i}] {source_label} ({ticker_label}, relevance={similarity}):\n"
            f"{r['content']}"
        )

    return "\n\n".join(context_parts)


def retrieve_multi_ticker(
    query:   str,
    tickers: list,
    top_k:   int = 3
) -> dict:
    # Retrieves context for multiple tickers simultaneously
    # Used for comparative queries like "compare AAPL vs MSFT"
    results = {}

    for ticker in tickers:
        ticker_results = retrieve(query, ticker=ticker, top_k=top_k)
        if ticker_results:
            results[ticker] = ticker_results

    return results


def test_retrieval():
    # Tests the retrieval pipeline with sample queries
    test_queries = [
        "Why is NVDA stock moving today?",
        "What is the sentiment for Apple?",
        "Are there any anomalies in Tesla recently?",
        "What do analysts say about Microsoft earnings?"
    ]

    print("Testing retrieval pipeline...")
    print("=" * 50)

    for query in test_queries:
        print(f"\nQuery: {query}")
        ticker  = extract_ticker(query)
        results = retrieve(query, ticker=ticker, top_k=3)

        print(f"Detected ticker: {ticker}")
        print(f"Results found : {len(results)}")

        for r in results[:2]:
            print(f"  [{r['similarity']:.3f}] {r['content'][:100]}...")