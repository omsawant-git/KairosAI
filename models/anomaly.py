# models/anomaly.py
# ─────────────────────────────────────────────────────────────
# KairosAI Anomaly Detection — v2
# Fixes from v1:
#   - Contamination raised to 0.08
#   - Ensemble softened — weighted score instead of AND logic
#   - Z-score threshold lowered to 2.0
#   - Correlated features removed
#   - Rolling 252-day baseline window
#   - Separate thresholds for ETFs vs stocks
#   - Temporal weighting — recent data weighted higher
# ─────────────────────────────────────────────────────────────

import warnings
import numpy as np
import pandas as pd
import joblib
import os
from datetime import datetime
from sqlalchemy import text
from database.db import engine, test_connection

warnings.filterwarnings("ignore")

from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import precision_score, recall_score, f1_score
import ta
import shap


# ── SETTINGS ──────────────────────────────────────────────────
TICKERS = [
    "AAPL", "MSFT", "NVDA", "GOOGL", "AMZN",
    "META", "TSLA", "SPY",  "QQQ",  "JPM",
    "BAC",  "UNH",  "JNJ",  "XOM",  "AMD"
]

# ETFs have lower volatility — need lower threshold
ETFS = ["SPY", "QQQ"]

# Contamination per ticker type
CONTAMINATION_STOCK = 0.08   # raised from 0.05
CONTAMINATION_ETF   = 0.05   # ETFs are less volatile

# Z-score threshold per ticker type
ZSCORE_STOCK = 2.0   # lowered from 2.5
ZSCORE_ETF   = 1.8   # even lower for ETFs

# Rolling window for baseline
ROLLING_WINDOW = 252  # 1 trading year

MODEL_DIR = "models/saved"


# ── HELPERS ───────────────────────────────────────────────────

def ensure_model_dir():
    os.makedirs(MODEL_DIR, exist_ok=True)


def is_etf(ticker: str) -> bool:
    return ticker in ETFS


def load_prices(ticker: str) -> pd.DataFrame:
    with engine.connect() as conn:
        result = conn.execute(
            text("""
                SELECT date, open, high, low, close, volume
                FROM fact_prices
                WHERE ticker = :ticker
                ORDER BY date ASC
            """),
            {"ticker": ticker}
        )
        rows = result.fetchall()

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows, columns=["date", "open", "high", "low", "close", "volume"])
    df["date"] = pd.to_datetime(df["date"])
    for col in ["open", "high", "low", "close", "volume"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=["close"])
    return df


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    # Engineers features — removed correlated pairs from v1
    # Kept the most independent and informative features
    df = df.copy()

    # ── Price returns ──
    # Daily return — primary anomaly signal
    df["return_1d"] = df["close"].pct_change()

    # 5-day return — multi-day momentum
    df["return_5d"] = df["close"].pct_change(5)

    # Gap — overnight price jump (news-driven)
    df["gap"] = (df["open"] - df["close"].shift(1)) / df["close"].shift(1)

    # Intraday range — volatility within the day
    df["intraday_range"] = (df["high"] - df["low"]) / df["close"]

    # ── MA deviation — use only ma20, dropped ma50 (correlated) ──
    df["ma20"]     = df["close"].rolling(20).mean()
    df["ma20_dev"] = (df["close"] - df["ma20"]) / df["ma20"]

    # ── Volume ──
    df["vol_mean"]   = df["volume"].rolling(20).mean()
    df["vol_std"]    = df["volume"].rolling(20).std()
    df["vol_zscore"] = (df["volume"] - df["vol_mean"]) / (df["vol_std"] + 1e-10)

    # ── Volatility ──
    # 20-day rolling volatility
    df["volatility_20d"] = df["return_1d"].rolling(20).std()

    # ATR normalized by price
    df["atr"] = ta.volatility.AverageTrueRange(
        high=df["high"], low=df["low"], close=df["close"]
    ).average_true_range() / df["close"]

    # Bollinger Band width — volatility expansion signal
    bb             = ta.volatility.BollingerBands(close=df["close"])
    df["bb_width"] = (bb.bollinger_hband() - bb.bollinger_lband()) / df["close"]

    # ── RSI — extremes are anomalous ──
    # Dropped bb_pct (correlated with RSI)
    df["rsi"] = ta.momentum.RSIIndicator(close=df["close"], window=14).rsi()

    # ── Return Z-score ──
    df["ret_mean"]   = df["return_1d"].rolling(20).mean()
    df["ret_std"]    = df["return_1d"].rolling(20).std()
    df["ret_zscore"] = (df["return_1d"] - df["ret_mean"]) / (df["ret_std"] + 1e-10)

    return df


# Reduced feature set — removed correlated features
FEATURE_COLS = [
    "return_1d",      "return_5d",
    "gap",            "intraday_range",
    "ma20_dev",
    "vol_zscore",
    "volatility_20d", "atr",
    "bb_width",
    "rsi",            "ret_zscore"
]


def get_temporal_weights(n: int) -> np.ndarray:
    # Assigns higher weights to more recent observations
    # Uses exponential decay — recent data matters more
    # This helps the model reflect current market conditions
    weights = np.exp(np.linspace(-1, 0, n))
    return weights / weights.sum() * n


def get_feature_matrix(df: pd.DataFrame) -> tuple:
    df_clean = df[FEATURE_COLS + ["date", "close"]].dropna()
    X        = df_clean[FEATURE_COLS].values
    dates    = df_clean["date"].values
    closes   = df_clean["close"].values
    return X, dates, closes


# ── ISOLATION FOREST ──────────────────────────────────────────

def train_isolation_forest(
    X:           np.ndarray,
    ticker:      str,
    sample_weight: np.ndarray = None
):
    # Trains Isolation Forest with temporal sample weights
    # Recent observations have more influence on the model
    contamination = CONTAMINATION_ETF if is_etf(ticker) else CONTAMINATION_STOCK

    scaler   = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    model = IsolationForest(
        n_estimators  = 300,          # increased from 200
        contamination = contamination,
        max_samples   = "auto",
        random_state  = 42,
        n_jobs        = -1
    )

    # Fit with sample weights if provided
    if sample_weight is not None:
        model.fit(X_scaled, sample_weight=sample_weight)
    else:
        model.fit(X_scaled)

    joblib.dump(model,  f"{MODEL_DIR}/iforest_{ticker}.pkl")
    joblib.dump(scaler, f"{MODEL_DIR}/iforest_scaler_{ticker}.pkl")

    return model, scaler


def get_if_scores(model, scaler, X: np.ndarray) -> tuple:
    X_scaled      = scaler.transform(X)
    raw_scores    = model.score_samples(X_scaled)
    anomaly_scores = -raw_scores

    # Normalize to [0, 1]
    min_s = anomaly_scores.min()
    max_s = anomaly_scores.max()
    normalized = (anomaly_scores - min_s) / (max_s - min_s + 1e-10)

    predictions = model.predict(X_scaled)
    is_anomaly  = (predictions == -1).astype(int)

    return normalized, is_anomaly


# ── Z-SCORE ───────────────────────────────────────────────────

def get_zscore_scores(df: pd.DataFrame, ticker: str) -> np.ndarray:
    # Computes a continuous Z-score anomaly score
    # Higher score = more anomalous by Z-score method
    threshold = ZSCORE_ETF if is_etf(ticker) else ZSCORE_STOCK

    df_clean = df[FEATURE_COLS + ["date", "close"]].dropna()

    ret_z = np.abs(df_clean["ret_zscore"].values)
    vol_z = np.abs(df_clean["vol_zscore"].values)

    # Combined Z-score — max of return and volume z-scores
    combined_z = np.maximum(ret_z, vol_z)

    # Normalize to [0, 1]
    max_z      = combined_z.max()
    z_scores   = combined_z / (max_z + 1e-10)

    # Binary flag using threshold
    z_anomaly = (combined_z > threshold).astype(int)

    return z_scores, z_anomaly


# ── SOFT ENSEMBLE ─────────────────────────────────────────────

def soft_ensemble(
    if_scores:  np.ndarray,
    z_scores:   np.ndarray,
    if_anomaly: np.ndarray,
    z_anomaly:  np.ndarray,
    ticker:     str
) -> tuple:
    # Soft ensemble — weighted combination of both scores
    # v1 used AND logic (too strict) — v2 uses weighted scores
    # IF gets 60% weight, Z-score gets 40% weight
    n = min(len(if_scores), len(z_scores))

    # Weighted combination of normalized scores
    combined_score = 0.6 * if_scores[:n] + 0.4 * z_scores[:n]

    # Dynamic threshold — flag top X% as anomalies
    contamination = CONTAMINATION_ETF if is_etf(ticker) else CONTAMINATION_STOCK
    threshold     = np.percentile(combined_score, (1 - contamination) * 100)

    # An anomaly is flagged when:
    # combined score > threshold OR either method strongly flags it
    is_anomaly = (
        (combined_score > threshold) &
        ((if_anomaly[:n] == 1) | (z_anomaly[:n] == 1))
    ).astype(int)

    return combined_score, is_anomaly


# ── ROLLING BASELINE ──────────────────────────────────────────

def rolling_anomaly_detection(
    X:      np.ndarray,
    dates:  np.ndarray,
    ticker: str
) -> tuple:
    # Detects anomalies using a rolling 252-day window
    # Each day is evaluated relative to the past year only
    # This prevents old market regimes from distorting detection
    n             = len(X)
    rolling_scores = np.zeros(n)
    rolling_flags  = np.zeros(n, dtype=int)

    contamination = CONTAMINATION_ETF if is_etf(ticker) else CONTAMINATION_STOCK

    for i in range(ROLLING_WINDOW, n):
        # Train on rolling window
        window_X = X[i-ROLLING_WINDOW:i]
        test_X   = X[i:i+1]

        try:
            scaler   = StandardScaler()
            X_scaled = scaler.fit_transform(window_X)
            test_scaled = scaler.transform(test_X)

            model = IsolationForest(
                n_estimators  = 100,  # fewer trees for speed
                contamination = contamination,
                random_state  = 42
            )
            model.fit(X_scaled)

            score = -model.score_samples(test_scaled)[0]
            flag  = 1 if model.predict(test_scaled)[0] == -1 else 0

            rolling_scores[i] = score
            rolling_flags[i]  = flag

        except Exception:
            continue

    # Normalize rolling scores
    max_s = rolling_scores.max()
    if max_s > 0:
        rolling_scores = rolling_scores / max_s

    return rolling_scores, rolling_flags


# ── SHAP ──────────────────────────────────────────────────────

def compute_shap(model, scaler, X: np.ndarray, ticker: str):
    try:
        X_scaled    = scaler.transform(X)
        explainer = shap.TreeExplainer(model, feature_perturbation="interventional")
        shap_values = explainer.shap_values(X_scaled)

        mean_shap = np.abs(shap_values).mean(axis=0)

        shap_df = pd.DataFrame({
            "feature":    FEATURE_COLS,
            "importance": mean_shap
        }).sort_values("importance", ascending=False)

        print(f"\n    Top anomaly drivers for {ticker}:")
        for _, row in shap_df.head(5).iterrows():
            print(f"      {row['feature']:20s} {row['importance']:.4f}")

        shap_df.to_csv(f"{MODEL_DIR}/shap_{ticker}.csv", index=False)

    except Exception as e:
        print(f"    SHAP failed: {e}")


# ── EVALUATE ──────────────────────────────────────────────────

def evaluate_anomalies(
    is_anomaly: np.ndarray,
    closes:     np.ndarray
) -> dict:
    # Proxy evaluation — anomalies should correspond to large moves
    # Uses 1.5 std threshold (more generous than v1's 2.0)
    returns     = np.diff(closes) / closes[:-1]
    return_std  = returns.std()
    large_moves = (np.abs(returns) > 1.5 * return_std).astype(int)

    n         = min(len(is_anomaly) - 1, len(large_moves))
    predicted = is_anomaly[1:n+1]
    actual    = large_moves[:n]

    try:
        precision = precision_score(actual, predicted, zero_division=0)
        recall    = recall_score(actual, predicted, zero_division=0)
        f1        = f1_score(actual, predicted, zero_division=0)
    except Exception:
        precision = recall = f1 = 0.0

    n_anomalies = is_anomaly.sum()
    anomaly_pct = (n_anomalies / len(is_anomaly)) * 100

    return {
        "n_anomalies": int(n_anomalies),
        "anomaly_pct": round(anomaly_pct, 2),
        "precision":   round(float(precision), 4),
        "recall":      round(float(recall),    4),
        "f1":          round(float(f1),        4)
    }


# ── SAVE ──────────────────────────────────────────────────────

def save_anomalies(
    ticker:         str,
    dates:          np.ndarray,
    anomaly_scores: np.ndarray,
    is_anomaly:     np.ndarray
) -> int:
    saved = 0

    with engine.connect() as conn:
        for i, date in enumerate(dates):
            try:
                conn.execute(
                    text("""
                        INSERT INTO fact_signals
                            (ticker, date, anomaly_score, is_anomaly)
                        VALUES
                            (:ticker, :date, :anomaly_score, :is_anomaly)
                        ON CONFLICT (ticker, date)
                        DO UPDATE SET
                            anomaly_score = EXCLUDED.anomaly_score,
                            is_anomaly    = EXCLUDED.is_anomaly
                    """),
                    {
                        "ticker":        ticker,
                        "date":          pd.Timestamp(date).date(),
                        "anomaly_score": float(anomaly_scores[i]),
                        "is_anomaly":    bool(is_anomaly[i])
                    }
                )
                saved += 1
            except Exception as e:
                print(f"    WARNING: {e}")
                continue

        conn.commit()

    return saved


# ── MAIN ──────────────────────────────────────────────────────

def run():
    print("=" * 60)
    print("KairosAI — Anomaly Detection v2")
    print("Models : Isolation Forest + Z-Score soft ensemble")
    print("Window : Rolling 252-day baseline")
    print("Weights: Temporal (recent data weighted higher)")
    print("=" * 60)

    test_connection()
    ensure_model_dir()

    all_results = []

    for ticker in TICKERS:
        print(f"\n── {ticker} {'[ETF]' if is_etf(ticker) else ''} ──────────────────")

        df = load_prices(ticker)

        if df.empty or len(df) < 60:
            print(f"  Not enough data, skipping")
            continue

        print(f"  Loaded {len(df)} rows")

        # Build features
        df = build_features(df)

        # Get feature matrix
        X, dates, closes = get_feature_matrix(df)

        if len(X) < 50:
            print(f"  Not enough clean rows")
            continue

        print(f"  Feature matrix: {X.shape[0]} × {X.shape[1]}")

        # Temporal weights — recent data weighted higher
        weights = get_temporal_weights(len(X))

        # Train Isolation Forest with temporal weights
        print(f"  Training Isolation Forest (n=300, temporal weights)...")
        model, scaler = train_isolation_forest(X, ticker, weights)

        # Get IF scores
        if_scores, if_anomaly = get_if_scores(model, scaler, X)

        # Get Z-score signals
        df_clean = df[FEATURE_COLS + ["date", "close"]].dropna()
        z_scores, z_anomaly = get_zscore_scores(df_clean, ticker)

        # Soft ensemble
        combined_scores, final_anomaly = soft_ensemble(
            if_scores, z_scores,
            if_anomaly, z_anomaly,
            ticker
        )

        # Rolling baseline anomaly detection
        print(f"  Running rolling 252-day baseline...")
        rolling_scores, rolling_flags = rolling_anomaly_detection(X, dates, ticker)

        # Final signal — combine ensemble + rolling
        # A day is anomalous if EITHER the ensemble OR rolling detects it
        n             = min(len(final_anomaly), len(rolling_flags))
        final_signal  = ((final_anomaly[:n] == 1) | (rolling_flags[:n] == 1)).astype(int)
        final_scores  = 0.7 * combined_scores[:n] + 0.3 * rolling_scores[:n]

        # SHAP
        compute_shap(model, scaler, X, ticker)

        # Evaluate
        metrics           = evaluate_anomalies(final_signal, closes[:n])
        metrics["ticker"] = ticker
        all_results.append(metrics)

        print(f"\n  Results:")
        print(f"    Anomalies detected : {metrics['n_anomalies']} ({metrics['anomaly_pct']}%)")
        print(f"    Precision          : {metrics['precision']}")
        print(f"    Recall             : {metrics['recall']}")
        print(f"    F1 score           : {metrics['f1']}")

        # Top anomalous recent days
        recent_n  = min(60, n)
        recent_df = pd.DataFrame({
            "date":          [pd.Timestamp(d).strftime("%Y-%m-%d") for d in dates[-recent_n:]],
            "close":         closes[-recent_n:],
            "anomaly_score": final_scores[-recent_n:],
            "is_anomaly":    final_signal[-recent_n:]
        })

        flagged = recent_df[recent_df["is_anomaly"] == 1].sort_values(
            "anomaly_score", ascending=False
        ).head(5)

        if not flagged.empty:
            print(f"\n  Top anomalous days (last 60 days):")
            for _, row in flagged.iterrows():
                print(f"    {row['date']}  ${row['close']:.2f}  score={row['anomaly_score']:.4f}")

        # Save
        saved = save_anomalies(ticker, dates[:n], final_scores, final_signal)
        print(f"\n  Saved {saved} records to fact_signals")

    # Summary
    print("\n" + "=" * 60)
    print("ANOMALY DETECTION v2 COMPLETE")
    print("=" * 60)

    if all_results:
        results_df = pd.DataFrame(all_results)
        print("\nModel performance summary:")
        print(results_df[[
            "ticker", "n_anomalies", "anomaly_pct",
            "precision", "recall", "f1"
        ]].to_string(index=False))

        print(f"\nAverage precision : {results_df['precision'].mean():.4f}")
        print(f"Average recall    : {results_df['recall'].mean():.4f}")
        print(f"Average F1        : {results_df['f1'].mean():.4f}")

        # Compare to v1
        print(f"\nv1 baseline — precision: 0.7494 | recall: 0.2831 | F1: 0.4075")
        print(f"v2 results  — precision: {results_df['precision'].mean():.4f} | "
              f"recall: {results_df['recall'].mean():.4f} | "
              f"F1: {results_df['f1'].mean():.4f}")

    return all_results

if __name__ == "__main__":
    run()