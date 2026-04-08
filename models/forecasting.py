# models/forecasting.py
# ─────────────────────────────────────────────────────────────
# Improved ARIMA + Prophet ensemble for 5-day price forecasting
# v3 fixes:
#   - Metrics computed on ensemble, not ARIMA alone
#   - ARIMA failure handled gracefully
#   - Holdout extended to 120 days
#   - auto_arima constrained to avoid (0,1,0) random walk
#   - Directional accuracy computed on ensemble predictions
# ─────────────────────────────────────────────────────────────

import os
import warnings
import joblib
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from sqlalchemy import text
from database.db import engine, test_connection

warnings.filterwarnings("ignore")

import logging
logging.getLogger("prophet").setLevel(logging.WARNING)
logging.getLogger("cmdstanpy").setLevel(logging.WARNING)

from statsmodels.tsa.stattools import adfuller
from pmdarima import auto_arima
from prophet import Prophet
from prophet.make_holidays import make_holidays_df
from sklearn.metrics import r2_score


# ── SETTINGS ──────────────────────────────────────────────────
TICKERS = [
    "AAPL", "MSFT", "NVDA", "GOOGL", "AMZN",
    "META", "TSLA", "SPY",  "QQQ",  "JPM",
    "BAC",  "UNH",  "JNJ",  "XOM",  "AMD"
]
FORECAST_DAYS = 5
HOLDOUT_DAYS  = 120    # extended from 60 — more stable evaluation
MODEL_DIR     = "models/saved"
MIN_ROWS      = 150


# ── HELPERS ───────────────────────────────────────────────────

def ensure_model_dir():
    os.makedirs(MODEL_DIR, exist_ok=True)


def load_prices(ticker: str) -> pd.DataFrame:
    with engine.connect() as conn:
        result = conn.execute(
            text("""
                SELECT date, close, volume
                FROM fact_prices
                WHERE ticker = :ticker
                ORDER BY date ASC
            """),
            {"ticker": ticker}
        )
        rows = result.fetchall()

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows, columns=["date", "close", "volume"])
    df["date"]   = pd.to_datetime(df["date"])
    df["close"]  = df["close"].astype(float)
    df["volume"] = df["volume"].astype(float)
    df = df.dropna(subset=["close"])
    return df


def clip_outliers(series: pd.Series) -> pd.Series:
    # Clips extreme values using IQR — prevents earnings spikes
    # from distorting model training
    Q1  = series.quantile(0.25)
    Q3  = series.quantile(0.75)
    IQR = Q3 - Q1
    return series.clip(lower=Q1 - 3*IQR, upper=Q3 + 3*IQR)


def check_stationarity(series: pd.Series) -> bool:
    # ADF test — returns True if stationary (p < 0.05)
    try:
        return adfuller(series.dropna())[1] < 0.05
    except Exception:
        return False


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    # Adds technical indicators used as Prophet regressors
    df = df.copy()
    # 20-day moving average — smooths noise
    df["ma20"] = df["close"].rolling(window=20).mean()
    # Normalized volume — high volume = big moves coming
    df["vol_norm"] = (
        df["volume"] / df["volume"].rolling(window=20).mean()
    ).fillna(1.0)
    # Daily return — captures momentum
    df["daily_return"] = df["close"].pct_change().fillna(0)
    return df


# ── ARIMA ─────────────────────────────────────────────────────

def train_arima(series: pd.Series, ticker: str):
    # auto_arima with constraints to avoid random walk (0,1,0)
    # start_p=2 forces at least some autoregressive component
    model_path = f"{MODEL_DIR}/arima_{ticker}.pkl"

    try:
        is_stationary = check_stationarity(series)
        d = 0 if is_stationary else 1

        model = auto_arima(
            series,
            d=d,
            start_p=2, max_p=6,   # enforce minimum AR order
            start_q=0, max_q=3,
            start_P=0, max_P=0,   # no seasonal component
            seasonal=False,
            stepwise=True,
            error_action="ignore",
            suppress_warnings=True,
            information_criterion="aic",
            trend="c"             # include constant term
        )

        # Reject random walk — if order is (0,1,0) fall back to (2,1,1)
        if model.order == (0, 1, 0) or model.order == (0, 0, 0):
            print(f"    auto_arima picked random walk, forcing (2,1,1)...")
            from statsmodels.tsa.arima.model import ARIMA as sm_ARIMA
            fallback = sm_ARIMA(series.values, order=(2,1,1)).fit()
            joblib.dump(fallback, model_path)
            return fallback, "statsmodels"

        joblib.dump(model, model_path)
        print(f"    Best ARIMA order: {model.order}")
        return model, "pmdarima"

    except Exception as e:
        print(f"    auto_arima failed: {e}")
        return None, None


def predict_arima(model, model_type: str) -> dict:
    # Handles both pmdarima and statsmodels fitted models
    try:
        if model_type == "pmdarima":
            forecast, conf_int = model.predict(
                n_periods=FORECAST_DAYS,
                return_conf_int=True
            )
            return {
                "prediction": float(forecast[-1]),
                "lower":      float(conf_int[-1][0]),
                "upper":      float(conf_int[-1][1])
            }
        else:
            # statsmodels fallback
            forecast = model.forecast(steps=FORECAST_DAYS)
            pred     = float(forecast[-1])
            return {
                "prediction": pred,
                "lower":      pred * 0.97,
                "upper":      pred * 1.03
            }
    except Exception as e:
        print(f"    ARIMA predict failed: {e}")
        return None


# ── PROPHET ───────────────────────────────────────────────────

def train_prophet(df: pd.DataFrame, ticker: str):
    model_path = f"{MODEL_DIR}/prophet_{ticker}.pkl"

    try:
        prophet_df = df[["date", "close", "ma20", "vol_norm"]].rename(
            columns={"date": "ds", "close": "y"}
        ).dropna()

        holidays = make_holidays_df(
            year_list=list(range(2019, 2027)),
            country="US"
        )

        model = Prophet(
            daily_seasonality=False,
            weekly_seasonality=True,
            yearly_seasonality=True,
            changepoint_prior_scale=0.05,
            holidays=holidays,
            interval_width=0.95
        )
        model.add_regressor("ma20")
        model.add_regressor("vol_norm")
        model.fit(prophet_df)

        joblib.dump(model, model_path)
        return model, prophet_df

    except Exception as e:
        print(f"    Prophet training failed: {e}")
        return None, None


def predict_prophet(model, df: pd.DataFrame) -> dict:
    try:
        future             = model.make_future_dataframe(periods=FORECAST_DAYS)
        future["ma20"]     = df["ma20"].iloc[-1]
        future["vol_norm"] = df["vol_norm"].iloc[-1]

        forecast = model.predict(future)
        last     = forecast.iloc[-1]

        return {
            "prediction": float(last["yhat"]),
            "lower":      float(last["yhat_lower"]),
            "upper":      float(last["yhat_upper"])
        }
    except Exception as e:
        print(f"    Prophet predict failed: {e}")
        return None


# ── ENSEMBLE ──────────────────────────────────────────────────

def weighted_ensemble(
    arima_pred:   dict,
    prophet_pred: dict,
    arima_rmse:   float,
    prophet_rmse: float
) -> dict:
    # Weights each model inversely by its holdout RMSE
    # Better model gets higher weight automatically
    if arima_pred is None and prophet_pred is None:
        return None
    if arima_pred is None:
        return prophet_pred
    if prophet_pred is None:
        return arima_pred

    arima_rmse   = max(arima_rmse,   0.0001)
    prophet_rmse = max(prophet_rmse, 0.0001)

    w_arima   = (1 / arima_rmse)
    w_prophet = (1 / prophet_rmse)
    total     = w_arima + w_prophet
    w_arima  /= total
    w_prophet /= total

    print(f"    Weights — ARIMA: {w_arima:.2f}, Prophet: {w_prophet:.2f}")

    return {
        "prediction": round(w_arima * arima_pred["prediction"] + w_prophet * prophet_pred["prediction"], 4),
        "lower":      round(w_arima * arima_pred["lower"]      + w_prophet * prophet_pred["lower"],      4),
        "upper":      round(w_arima * arima_pred["upper"]      + w_prophet * prophet_pred["upper"],      4),
        "w_arima":    round(w_arima,   4),
        "w_prophet":  round(w_prophet, 4)
    }


# ── EVALUATION ────────────────────────────────────────────────

def evaluate_on_holdout(df: pd.DataFrame) -> tuple:
    # Evaluates both models on holdout set
    # Returns (arima_rmse, prophet_rmse, ensemble_predictions, actuals)
    split    = len(df) - HOLDOUT_DAYS
    train_df = df.iloc[:split].copy()
    test_df  = df.iloc[split:].copy()

    train_close = train_df["close"]
    test_close  = test_df["close"].values

    # ARIMA on holdout
    arima_rmse = 999.0
    try:
        eval_arima, eval_type = train_arima(train_close, "_eval")
        if eval_arima and eval_type == "pmdarima":
            arima_preds = eval_arima.predict(n_periods=len(test_close))
        elif eval_arima:
            arima_preds = eval_arima.forecast(steps=len(test_close))
        else:
            arima_preds = np.full(len(test_close), train_close.mean())

        arima_rmse = float(np.sqrt(np.mean((test_close - arima_preds) ** 2)))
    except Exception as e:
        print(f"    ARIMA eval failed: {e}")
        arima_preds = np.full(len(test_close), train_close.mean())

    # Prophet on holdout
    prophet_rmse = 999.0
    try:
        prophet_train = train_df[["date", "close", "ma20", "vol_norm"]].rename(
            columns={"date": "ds", "close": "y"}
        ).dropna()

        eval_prophet = Prophet(
            daily_seasonality=False,
            weekly_seasonality=True,
            yearly_seasonality=True,
            changepoint_prior_scale=0.05,
            interval_width=0.95
        )
        eval_prophet.add_regressor("ma20")
        eval_prophet.add_regressor("vol_norm")
        eval_prophet.fit(prophet_train)

        future             = eval_prophet.make_future_dataframe(periods=len(test_df))
        future["ma20"]     = prophet_train["ma20"].iloc[-1]
        future["vol_norm"] = prophet_train["vol_norm"].iloc[-1]

        forecast      = eval_prophet.predict(future)
        prophet_preds = forecast["yhat"].iloc[-len(test_df):].values

        prophet_rmse = float(np.sqrt(np.mean(
            (test_close[:len(prophet_preds)] - prophet_preds) ** 2
        )))
    except Exception as e:
        print(f"    Prophet eval failed: {e}")
        prophet_preds = np.full(len(test_close), train_close.mean())

    # Ensemble predictions on holdout
    a_rmse = max(arima_rmse,   0.0001)
    p_rmse = max(prophet_rmse, 0.0001)
    w_a    = (1/a_rmse) / ((1/a_rmse) + (1/p_rmse))
    w_p    = 1 - w_a

    n              = min(len(arima_preds), len(prophet_preds), len(test_close))
    ensemble_preds = w_a * arima_preds[:n] + w_p * prophet_preds[:n]
    actuals        = test_close[:n]

    return arima_rmse, prophet_rmse, ensemble_preds, actuals


def compute_metrics(actuals: np.ndarray, predictions: np.ndarray) -> dict:
    # Full suite of evaluation metrics on ensemble predictions
    rmse = float(np.sqrt(np.mean((actuals - predictions) ** 2)))
    mae  = float(np.mean(np.abs(actuals - predictions)))
    mape = float(np.mean(np.abs((actuals - predictions) / actuals)) * 100)
    smape = float(np.mean(
        2 * np.abs(actuals - predictions) /
        (np.abs(actuals) + np.abs(predictions))
    ) * 100)

    try:
        r2 = float(r2_score(actuals, predictions))
    except Exception:
        r2 = 0.0

    # Directional accuracy — most important for trading signals
    actual_dir = np.sign(np.diff(actuals))
    pred_dir   = np.sign(np.diff(predictions))
    dir_acc    = float(np.mean(actual_dir == pred_dir) * 100)

    return {
        "rmse":            round(rmse,   4),
        "mae":             round(mae,    4),
        "mape":            round(mape,   4),
        "smape":           round(smape,  4),
        "r2":              round(r2,     4),
        "directional_acc": round(dir_acc, 2)
    }


# ── SAVE ──────────────────────────────────────────────────────

def save_prediction(ticker: str, result: dict):
    if not result or result.get("prediction") is None:
        return

    target_date = (datetime.today() + timedelta(days=FORECAST_DAYS)).date()

    with engine.connect() as conn:
        conn.execute(
            text("""
                INSERT INTO fact_signals
                    (ticker, date, predicted_close)
                VALUES
                    (:ticker, :date, :predicted_close)
                ON CONFLICT (ticker, date)
                DO UPDATE SET predicted_close = EXCLUDED.predicted_close
            """),
            {
                "ticker":          ticker,
                "date":            target_date,
                "predicted_close": result["prediction"]
            }
        )
        conn.commit()


# ── MAIN ──────────────────────────────────────────────────────

def run():
    print("=" * 55)
    print("KairosAI — Forecasting v3 (auto_arima + Prophet)")
    print(f"Forecast horizon : {FORECAST_DAYS} days")
    print(f"Holdout period   : {HOLDOUT_DAYS} days")
    print("=" * 55)

    test_connection()
    ensure_model_dir()

    all_metrics = []

    for ticker in TICKERS:
        print(f"\n── {ticker} ──────────────────────────────")

        df = load_prices(ticker)

        if df.empty or len(df) < MIN_ROWS:
            print(f"  Not enough data ({len(df)} rows), skipping")
            continue

        print(f"  Loaded {len(df)} rows")

        # Prepare data
        df["close"] = clip_outliers(df["close"])
        df          = add_features(df)

        # Evaluate on holdout — get RMSE per model + ensemble predictions
        print(f"  Evaluating on {HOLDOUT_DAYS}-day holdout...")
        arima_rmse, prophet_rmse, ensemble_preds, actuals = evaluate_on_holdout(df)
        print(f"  Holdout RMSE — ARIMA: ${arima_rmse:.2f} | Prophet: ${prophet_rmse:.2f}")

        # Compute metrics on ensemble predictions
        metrics            = compute_metrics(actuals, ensemble_preds)
        metrics["ticker"]  = ticker
        all_metrics.append(metrics)

        print(f"  Ensemble metrics — RMSE: ${metrics['rmse']} | "
              f"MAPE: {metrics['mape']}% | R²: {metrics['r2']} | "
              f"Dir. Acc: {metrics['directional_acc']}%")

        # Train on full data
        print(f"  Training on full dataset...")
        arima_model, arima_type      = train_arima(df["close"], ticker)
        prophet_model, prophet_df_   = train_prophet(df, ticker)

        # Predict
        arima_result   = predict_arima(arima_model, arima_type) if arima_model else None
        prophet_result = predict_prophet(prophet_model, df)     if prophet_model else None

        # Weighted ensemble
        final = weighted_ensemble(
            arima_result, prophet_result,
            arima_rmse,   prophet_rmse
        )

        if final:
            current_price = float(df["close"].iloc[-1])
            direction     = "UP"   if final["prediction"] > current_price else "DOWN"
            change_pct    = ((final["prediction"] - current_price) / current_price) * 100

            print(f"  Current : ${current_price:.2f}")
            print(f"  Forecast: ${final['prediction']:.2f} "
                  f"({direction} {abs(change_pct):.2f}%)")
            print(f"  CI 95%  : [${final['lower']:.2f} — ${final['upper']:.2f}]")

        save_prediction(ticker, final)

    # Summary table
    print("\n" + "=" * 55)
    print("FORECASTING COMPLETE")
    print("=" * 55)

    if all_metrics:
        metrics_df = pd.DataFrame(all_metrics)
        print("\nEnsemble model performance on holdout:")
        print(metrics_df[[
            "ticker", "rmse", "mae", "mape", "r2", "directional_acc"
        ]].to_string(index=False))

        avg_dir = metrics_df["directional_acc"].mean()
        avg_r2  = metrics_df["r2"].mean()
        print(f"\nAverage directional accuracy : {avg_dir:.2f}%")
        print(f"Average R²                   : {avg_r2:.4f}")
        print(f"Random baseline              : 50.00%")

        above_baseline = (metrics_df["directional_acc"] > 50).sum()
        print(f"Tickers above random baseline: {above_baseline}/{len(metrics_df)}")

    return all_metrics


if __name__ == "__main__":
    run()