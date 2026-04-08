-- ─────────────────────────────────────────────────────────────
-- KairosAI Database Schema
-- ─────────────────────────────────────────────────────────────

-- Enable the pgvector extension so we can store ML embeddings
-- as vector columns (used in the RAG pipeline)
CREATE EXTENSION IF NOT EXISTS vector;

-- Enable UUID generation so we can use uuid_generate_v4()
-- as primary keys instead of plain integers
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";


-- ─────────────────────────────────────────────────────────────
-- AUTH TABLES
-- These three tables handle user accounts and watchlists
-- ─────────────────────────────────────────────────────────────

-- Stores one row per registered user
-- hashed_password: we never store plain text passwords, only bcrypt hashes
CREATE TABLE IF NOT EXISTS users (
    id              UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    email           VARCHAR(255) UNIQUE NOT NULL,
    hashed_password TEXT NOT NULL,
    full_name       VARCHAR(255),
    created_at      TIMESTAMPTZ DEFAULT NOW()
);

-- Tracks active login sessions
-- When a user logs in we store a hash of their JWT token here
-- When they log out we delete the row
CREATE TABLE IF NOT EXISTS user_sessions (
    id          UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id     UUID REFERENCES users(id) ON DELETE CASCADE,
    token_hash  TEXT NOT NULL,
    created_at  TIMESTAMPTZ DEFAULT NOW(),
    expires_at  TIMESTAMPTZ NOT NULL
);

-- Each user can have their own list of tickers they want to track
-- UNIQUE(user_id, ticker) prevents duplicate tickers per user
CREATE TABLE IF NOT EXISTS user_watchlists (
    id          UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id     UUID REFERENCES users(id) ON DELETE CASCADE,
    ticker      VARCHAR(20) NOT NULL,
    added_at    TIMESTAMPTZ DEFAULT NOW(),
    UNIQUE(user_id, ticker)
);


-- ─────────────────────────────────────────────────────────────
-- DIMENSION TABLES
-- These are lookup/reference tables used to enrich the facts
-- ─────────────────────────────────────────────────────────────

-- Master list of all tickers we track
-- sector and industry come from yfinance metadata
CREATE TABLE IF NOT EXISTS dim_ticker (
    id          SERIAL PRIMARY KEY,
    ticker      VARCHAR(20) UNIQUE NOT NULL,
    company     VARCHAR(255),
    sector      VARCHAR(100),
    industry    VARCHAR(100)
);

-- Calendar table — one row per date
-- is_trading_day lets us skip weekends and holidays in queries
CREATE TABLE IF NOT EXISTS dim_date (
    date_id         DATE PRIMARY KEY,
    year            INT,
    month           INT,
    day             INT,
    weekday         VARCHAR(10),
    is_trading_day  BOOLEAN DEFAULT TRUE
);


-- ─────────────────────────────────────────────────────────────
-- FACT TABLES
-- These are the core data tables — high volume, append-only
-- ─────────────────────────────────────────────────────────────

-- Daily OHLCV price data pulled from yfinance
-- UNIQUE(ticker, date) prevents duplicate rows for the same day
CREATE TABLE IF NOT EXISTS fact_prices (
    id          SERIAL PRIMARY KEY,
    ticker      VARCHAR(20) NOT NULL,
    date        DATE NOT NULL,
    open        NUMERIC(12,4),
    high        NUMERIC(12,4),
    low         NUMERIC(12,4),
    close       NUMERIC(12,4),
    volume      BIGINT,
    created_at  TIMESTAMPTZ DEFAULT NOW(),
    UNIQUE(ticker, date)
);

-- Real-time price ticks streamed from Alpaca WebSocket
-- This table gets written to every few seconds while the stream runs
-- Used to power the live price display on the dashboard
CREATE TABLE IF NOT EXISTS live_quotes (
    id          SERIAL PRIMARY KEY,
    ticker      VARCHAR(20) NOT NULL,
    price       NUMERIC(12,4),
    volume      BIGINT,
    timestamp   TIMESTAMPTZ DEFAULT NOW()
);

-- ML model outputs — one row per ticker per day
-- predicted_close: ARIMA+Prophet 5-day forecast
-- anomaly_score: Isolation Forest score (lower = more anomalous)
-- is_anomaly: boolean flag when score crosses threshold
-- sentiment_score: FinBERT score from -1 (negative) to +1 (positive)
CREATE TABLE IF NOT EXISTS fact_signals (
    id               SERIAL PRIMARY KEY,
    ticker           VARCHAR(20) NOT NULL,
    date             DATE NOT NULL,
    predicted_close  NUMERIC(12,4),
    anomaly_score    NUMERIC(8,4),
    is_anomaly       BOOLEAN DEFAULT FALSE,
    sentiment_score  NUMERIC(5,4),
    created_at       TIMESTAMPTZ DEFAULT NOW(),
    UNIQUE(ticker, date)
);

-- News headlines fetched from Finnhub per ticker
-- These get chunked and embedded into document_chunks for the RAG pipeline
CREATE TABLE IF NOT EXISTS fact_news (
    id           SERIAL PRIMARY KEY,
    ticker       VARCHAR(20),
    headline     TEXT NOT NULL,
    source       VARCHAR(255),
    url          TEXT,
    published_at TIMESTAMPTZ,
    created_at   TIMESTAMPTZ DEFAULT NOW()
);

-- Macroeconomic data from FRED API
-- series_id examples: GDP, CPIAUCSL, FEDFUNDS, UNRATE
-- UNIQUE(series_id, date) prevents duplicate entries per series per day
CREATE TABLE IF NOT EXISTS fact_macro (
    id          SERIAL PRIMARY KEY,
    series_id   VARCHAR(50) NOT NULL,
    date        DATE NOT NULL,
    value       NUMERIC(16,6),
    created_at  TIMESTAMPTZ DEFAULT NOW(),
    UNIQUE(series_id, date)
);


-- ─────────────────────────────────────────────────────────────
-- RAG TABLE
-- Stores chunked documents with their vector embeddings
-- ─────────────────────────────────────────────────────────────

-- Each row is one chunk of text (news article, SEC filing paragraph etc.)
-- embedding: 384-dimension vector from sentence-transformers all-MiniLM-L6-v2
-- The agent queries this table using cosine similarity to find
-- relevant context before answering a question
CREATE TABLE IF NOT EXISTS document_chunks (
    id          SERIAL PRIMARY KEY,
    ticker      VARCHAR(20),
    source      VARCHAR(50),
    content     TEXT NOT NULL,
    embedding   vector(384),
    created_at  TIMESTAMPTZ DEFAULT NOW()
);


-- ─────────────────────────────────────────────────────────────
-- INDEXES
-- Speed up the most common query patterns
-- ─────────────────────────────────────────────────────────────

-- Most queries filter by ticker + date range so index both together
CREATE INDEX IF NOT EXISTS idx_prices_ticker_date  ON fact_prices(ticker, date);
CREATE INDEX IF NOT EXISTS idx_signals_ticker_date ON fact_signals(ticker, date);

-- News and chunks are almost always filtered by ticker
CREATE INDEX IF NOT EXISTS idx_news_ticker        ON fact_news(ticker);
CREATE INDEX IF NOT EXISTS idx_chunks_ticker      ON document_chunks(ticker);

-- Live quotes are queried by ticker to get the latest price
CREATE INDEX IF NOT EXISTS idx_live_quotes_ticker ON live_quotes(ticker);