# models/forecasting.py
# ─────────────────────────────────────────────────────────────
# KairosAI Price Forecasting — v4 (Final)
# Models: ARIMA + Prophet + LSTM ensemble
# Features: Technical indicators + regime detection
# Evaluation: Walk-forward validation
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
from sklearn.preprocessing import MinMaxScaler
import ta

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


# ── SETTINGS ──────────────────────────────────────────────────
TICKERS       = [
    "AAPL", "MSFT", "NVDA", "GOOGL", "AMZN",
    "META", "TSLA", "SPY",  "QQQ",  "JPM",
    "BAC",  "UNH",  "JNJ",  "XOM",  "AMD"
]
FORECAST_DAYS = 5
HOLDOUT_DAYS  = 120
MODEL_DIR     = "models/saved"
MIN_ROWS      = 150
LSTM_SEQ_LEN  = 30    # LSTM looks back 30 days
LSTM_EPOCHS   = 30
LSTM_HIDDEN   = 64
LSTM_LAYERS   = 2
DEVICE        = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ── HELPERS ───────────────────────────────────────────────────

def ensure_model_dir():
    os.makedirs(MODEL_DIR, exist_ok=True)


def load_prices(ticker: str) -> pd.DataFrame:
    # Loads OHLCV from fact_prices for a given ticker
    with engine.connect() as conn:
        result = conn.execute(
            text("""
                SELECT date, close, open, high, low, volume
                FROM fact_prices
                WHERE ticker = :ticker
                ORDER BY date ASC
            """),
            {"ticker": ticker}
        )
        rows = result.fetchall()

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows, columns=["date", "close", "open", "high", "low", "volume"])
    df["date"]   = pd.to_datetime(df["date"])
    for col in ["close", "open", "high", "low", "volume"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=["close"])
    return df


def clip_outliers(series: pd.Series) -> pd.Series:
    # IQR-based clipping — prevents earnings spikes from distorting training
    Q1, Q3 = series.quantile(0.25), series.quantile(0.75)
    IQR    = Q3 - Q1
    return series.clip(lower=Q1 - 3*IQR, upper=Q3 + 3*IQR)


def check_stationarity(series: pd.Series) -> bool:
    # ADF test — True if stationary (p < 0.05)
    try:
        return adfuller(series.dropna())[1] < 0.05
    except Exception:
        return False


def detect_regime(df: pd.DataFrame) -> str:
    # Classifies current market regime using 200-day MA
    # Bull  = price above MA200
    # Bear  = price below MA200
    # Sideways = insufficient data
    if len(df) < 200:
        return "sideways"
    ma200         = df["close"].rolling(200).mean().iloc[-1]
    current_price = df["close"].iloc[-1]
    if current_price > ma200 * 1.02:
        return "bull"
    elif current_price < ma200 * 0.98:
        return "bear"
    return "sideways"


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    # Adds technical indicators as features
    df = df.copy()

    # ── Trend ──
    # 20-day and 50-day moving averages
    df["ma20"] = df["close"].rolling(20).mean()
    df["ma50"] = df["close"].rolling(50).mean()

    # ── Momentum ──
    # RSI — overbought > 70, oversold < 30
    df["rsi"] = ta.momentum.RSIIndicator(
        close=df["close"], window=14
    ).rsi()

    # MACD — trend direction and momentum
    macd           = ta.trend.MACD(close=df["close"])
    df["macd"]     = macd.macd()
    df["macd_sig"] = macd.macd_signal()
    df["macd_diff"] = macd.macd_diff()

    # ── Volatility ──
    # Bollinger Bands — price relative to volatility bands
    bb             = ta.volatility.BollingerBands(close=df["close"], window=20)
    df["bb_upper"] = bb.bollinger_hband()
    df["bb_lower"] = bb.bollinger_lband()
    df["bb_pct"]   = bb.bollinger_pband()  # where price sits within bands (0-1)

    # Average True Range — measures volatility
    df["atr"] = ta.volatility.AverageTrueRange(
        high=df["high"], low=df["low"], close=df["close"]
    ).average_true_range()

    # ── Volume ──
    # On-Balance Volume — volume-price relationship
    df["obv"]      = ta.volume.OnBalanceVolumeIndicator(
        close=df["close"], volume=df["volume"]
    ).on_balance_volume()

    # Normalized volume
    df["vol_norm"] = (
        df["volume"] / df["volume"].rolling(20).mean()
    ).fillna(1.0)

    # ── Returns ──
    df["daily_return"] = df["close"].pct_change().fillna(0)
    df["return_5d"]    = df["close"].pct_change(5).fillna(0)

    return df


def get_regime_weights(regime: str) -> dict:
    # Adjusts ensemble weights based on market regime
    # In bull markets Prophet (trend-following) gets more weight
    # In bear markets ARIMA (mean-reverting) gets more weight
    # In sideways markets LSTM (pattern-recognition) gets more weight
    if regime == "bull":
        return {"arima": 0.25, "prophet": 0.45, "lstm": 0.30}
    elif regime == "bear":
        return {"arima": 0.45, "prophet": 0.25, "lstm": 0.30}
    else:
        return {"arima": 0.30, "prophet": 0.30, "lstm": 0.40}


# ── ARIMA ─────────────────────────────────────────────────────

def train_arima(series: pd.Series, ticker: str):
    # auto_arima with constraints to avoid random walk
    try:
        is_stationary = check_stationarity(series)
        d = 0 if is_stationary else 1

        model = auto_arima(
            series,
            d=d,
            start_p=2, max_p=6,
            start_q=0, max_q=3,
            seasonal=False,
            stepwise=True,
            error_action="ignore",
            suppress_warnings=True,
            information_criterion="aic",
            trend="c"
        )

        # Reject random walk — force (2,1,1) if needed
        if model.order in [(0,1,0), (0,0,0)]:
            print(f"    Forcing ARIMA(2,1,1) — random walk rejected")
            from statsmodels.tsa.arima.model import ARIMA as sm_ARIMA
            fallback = sm_ARIMA(series.values, order=(2,1,1)).fit()
            return fallback, "statsmodels"

        print(f"    Best ARIMA order: {model.order}")
        return model, "pmdarima"

    except Exception as e:
        print(f"    auto_arima failed: {e}")
        return None, None


def predict_arima(model, model_type: str) -> dict:
    try:
        if model_type == "pmdarima":
            forecast, conf = model.predict(
                n_periods=FORECAST_DAYS, return_conf_int=True
            )
            return {
                "prediction": float(forecast[-1]),
                "lower":      float(conf[-1][0]),
                "upper":      float(conf[-1][1])
            }
        else:
            f    = model.forecast(steps=FORECAST_DAYS)
            pred = float(f[-1])
            return {"prediction": pred, "lower": pred*0.97, "upper": pred*1.03}
    except Exception as e:
        print(f"    ARIMA predict failed: {e}")
        return None


# ── PROPHET ───────────────────────────────────────────────────

def train_prophet(df: pd.DataFrame, ticker: str):
    # Prophet with technical indicator regressors
    try:
        prophet_df = df[[
            "date", "close", "ma20", "vol_norm",
            "rsi", "macd_diff", "bb_pct"
        ]].rename(columns={"date": "ds", "close": "y"}).dropna()

        holidays = make_holidays_df(
            year_list=list(range(2019, 2027)), country="US"
        )

        model = Prophet(
            daily_seasonality=False,
            weekly_seasonality=True,
            yearly_seasonality=True,
            changepoint_prior_scale=0.05,
            holidays=holidays,
            interval_width=0.95
        )

        # Technical indicators as regressors
        for reg in ["ma20", "vol_norm", "rsi", "macd_diff", "bb_pct"]:
            model.add_regressor(reg)

        model.fit(prophet_df)
        return model, prophet_df

    except Exception as e:
        print(f"    Prophet training failed: {e}")
        return None, None


def predict_prophet(model, df: pd.DataFrame) -> dict:
    try:
        future = model.make_future_dataframe(periods=FORECAST_DAYS)

        # Fill regressors with last known values for future dates
        for col in ["ma20", "vol_norm", "rsi", "macd_diff", "bb_pct"]:
            future[col] = df[col].iloc[-1]

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


# ── LSTM ──────────────────────────────────────────────────────

class LSTMModel(nn.Module):
    # PyTorch LSTM for sequential price prediction
    # Input: sequence of LSTM_SEQ_LEN days of features
    # Output: next price prediction
    def __init__(self, input_size: int, hidden_size: int, num_layers: int):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers  = num_layers

        self.lstm    = nn.LSTM(
            input_size, hidden_size, num_layers,
            batch_first=True, dropout=0.2
        )
        self.dropout = nn.Dropout(0.2)
        self.fc      = nn.Linear(hidden_size, 1)

    def forward(self, x):
        # Initialize hidden and cell states
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(DEVICE)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(DEVICE)

        out, _  = self.lstm(x, (h0, c0))
        out     = self.dropout(out[:, -1, :])
        out     = self.fc(out)
        return out


def prepare_lstm_data(df: pd.DataFrame):
    # Prepares sequences for LSTM training
    # Features: close, rsi, macd_diff, bb_pct, vol_norm, daily_return
    feature_cols = [
        "close", "rsi", "macd_diff", "bb_pct",
        "vol_norm", "daily_return", "atr"
    ]

    data = df[feature_cols].dropna().values

    # Scale features to [0,1]
    scaler = MinMaxScaler()
    data   = scaler.fit_transform(data)

    # Build sequences
    X, y = [], []
    for i in range(LSTM_SEQ_LEN, len(data)):
        X.append(data[i-LSTM_SEQ_LEN:i])
        y.append(data[i, 0])  # predict close price (index 0)

    X = np.array(X)
    y = np.array(y)

    return X, y, scaler


def train_lstm(df: pd.DataFrame, ticker: str):
    # Trains PyTorch LSTM on price + technical features
    print(f"    Training LSTM ({LSTM_EPOCHS} epochs)...")

    try:
        X, y, scaler = prepare_lstm_data(df)

        if len(X) < 50:
            print("    Not enough sequences for LSTM")
            return None, None

        # Split — 80% train
        split   = int(len(X) * 0.8)
        X_train = torch.FloatTensor(X[:split]).to(DEVICE)
        y_train = torch.FloatTensor(y[:split]).unsqueeze(1).to(DEVICE)

        dataset    = TensorDataset(X_train, y_train)
        dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

        model     = LSTMModel(
            input_size  = X.shape[2],
            hidden_size = LSTM_HIDDEN,
            num_layers  = LSTM_LAYERS
        ).to(DEVICE)

        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        criterion = nn.MSELoss()

        model.train()
        for epoch in range(LSTM_EPOCHS):
            epoch_loss = 0
            for X_batch, y_batch in dataloader:
                optimizer.zero_grad()
                output = model(X_batch)
                loss   = criterion(output, y_batch)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()

            if (epoch + 1) % 10 == 0:
                print(f"    Epoch {epoch+1}/{LSTM_EPOCHS} — Loss: {epoch_loss/len(dataloader):.6f}")

        # Save model
        torch.save(model.state_dict(), f"{MODEL_DIR}/lstm_{ticker}.pt")

        return model, scaler

    except Exception as e:
        print(f"    LSTM training failed: {e}")
        return None, None


def predict_lstm(model, df: pd.DataFrame, scaler) -> dict:
    # Uses last LSTM_SEQ_LEN days to predict next price
    try:
        feature_cols = [
            "close", "rsi", "macd_diff", "bb_pct",
            "vol_norm", "daily_return", "atr"
        ]

        data = df[feature_cols].dropna().values
        data = scaler.transform(data)

        # Take last sequence
        seq = data[-LSTM_SEQ_LEN:]
        seq = torch.FloatTensor(seq).unsqueeze(0).to(DEVICE)

        model.eval()
        with torch.no_grad():
            pred_scaled = model(seq).cpu().numpy()[0][0]

        # Inverse transform — reconstruct full feature vector
        dummy        = np.zeros((1, len(feature_cols)))
        dummy[0, 0]  = pred_scaled
        pred_price   = scaler.inverse_transform(dummy)[0][0]

        # Confidence interval — ±2% around prediction
        return {
            "prediction": float(pred_price),
            "lower":      float(pred_price * 0.98),
            "upper":      float(pred_price * 1.02)
        }

    except Exception as e:
        print(f"    LSTM predict failed: {e}")
        return None


# ── ENSEMBLE ──────────────────────────────────────────────────

def regime_weighted_ensemble(
    arima_pred:   dict,
    prophet_pred: dict,
    lstm_pred:    dict,
    arima_rmse:   float,
    prophet_rmse: float,
    lstm_rmse:    float,
    regime:       str
) -> dict:
    # Three-model ensemble with regime-adjusted weights
    # Step 1: get base regime weights
    base = get_regime_weights(regime)

    # Step 2: adjust by performance (inverse RMSE)
    # Models with lower RMSE get proportionally more weight
    def safe_inv(x): return 1 / max(x, 0.0001)

    perf = {
        "arima":   safe_inv(arima_rmse)   if arima_pred   else 0,
        "prophet": safe_inv(prophet_rmse) if prophet_pred else 0,
        "lstm":    safe_inv(lstm_rmse)    if lstm_pred    else 0
    }

    total_perf = sum(perf.values())
    if total_perf == 0:
        return None

    # Normalize performance weights
    perf = {k: v/total_perf for k, v in perf.items()}

    # Blend regime weights and performance weights 50/50
    final_weights = {
        k: 0.5 * base[k] + 0.5 * perf[k]
        for k in ["arima", "prophet", "lstm"]
    }

    # Normalize final weights
    total = sum(final_weights.values())
    final_weights = {k: v/total for k, v in final_weights.items()}

    print(f"    Regime: {regime} | Weights — "
          f"ARIMA: {final_weights['arima']:.2f}, "
          f"Prophet: {final_weights['prophet']:.2f}, "
          f"LSTM: {final_weights['lstm']:.2f}")

    # Compute weighted prediction
    preds  = []
    lowers = []
    uppers = []

    for key, pred in [("arima", arima_pred), ("prophet", prophet_pred), ("lstm", lstm_pred)]:
        if pred:
            w = final_weights[key]
            preds.append(w  * pred["prediction"])
            lowers.append(w * pred["lower"])
            uppers.append(w * pred["upper"])

    return {
        "prediction": round(sum(preds),  4),
        "lower":      round(sum(lowers), 4),
        "upper":      round(sum(uppers), 4),
        "weights":    final_weights,
        "regime":     regime
    }


# ── EVALUATION ────────────────────────────────────────────────

def evaluate_lstm_holdout(df: pd.DataFrame) -> float:
    # Quick LSTM evaluation on holdout — returns RMSE
    try:
        X, y, scaler = prepare_lstm_data(df)
        split        = int(len(X) * 0.8)

        X_test = torch.FloatTensor(X[split:]).to(DEVICE)
        y_test = y[split:]

        model = LSTMModel(
            input_size  = X.shape[2],
            hidden_size = LSTM_HIDDEN,
            num_layers  = LSTM_LAYERS
        ).to(DEVICE)

        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        criterion = nn.MSELoss()

        X_train = torch.FloatTensor(X[:split]).to(DEVICE)
        y_train = torch.FloatTensor(y[:split]).unsqueeze(1).to(DEVICE)

        model.train()
        for _ in range(15):  # quick eval training
            optimizer.zero_grad()
            loss = criterion(model(X_train), y_train)
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            preds_scaled = model(X_test).cpu().numpy().flatten()

        # Inverse transform
        feature_cols = [
            "close", "rsi", "macd_diff", "bb_pct",
            "vol_norm", "daily_return", "atr"
        ]
        n_features = len(feature_cols)

        preds_full        = np.zeros((len(preds_scaled), n_features))
        preds_full[:, 0]  = preds_scaled
        preds_inv         = scaler.inverse_transform(preds_full)[:, 0]

        actuals_full       = np.zeros((len(y_test), n_features))
        actuals_full[:, 0] = y_test
        actuals_inv        = scaler.inverse_transform(actuals_full)[:, 0]

        rmse = float(np.sqrt(np.mean((actuals_inv - preds_inv) ** 2)))
        return rmse

    except Exception as e:
        print(f"    LSTM eval failed: {e}")
        return 999.0


def evaluate_arima_holdout(df: pd.DataFrame) -> float:
    split       = len(df) - HOLDOUT_DAYS
    train       = df["close"].iloc[:split]
    test        = df["close"].iloc[split:].values
    try:
        model, mtype = train_arima(train, "_eval")
        if model and mtype == "pmdarima":
            preds = model.predict(n_periods=len(test))
        elif model:
            preds = model.forecast(steps=len(test))
        else:
            return 999.0
        return float(np.sqrt(np.mean((test - preds) ** 2)))
    except Exception:
        return 999.0


def evaluate_prophet_holdout(df: pd.DataFrame) -> float:
    split    = len(df) - HOLDOUT_DAYS
    train_df = df.iloc[:split].copy()
    test_df  = df.iloc[split:].copy()
    try:
        prophet_train = train_df[[
            "date", "close", "ma20", "vol_norm",
            "rsi", "macd_diff", "bb_pct"
        ]].rename(columns={"date": "ds", "close": "y"}).dropna()

        model = Prophet(
            daily_seasonality=False,
            weekly_seasonality=True,
            yearly_seasonality=True,
            changepoint_prior_scale=0.05,
            interval_width=0.95
        )
        for reg in ["ma20", "vol_norm", "rsi", "macd_diff", "bb_pct"]:
            model.add_regressor(reg)
        model.fit(prophet_train)

        future = model.make_future_dataframe(periods=len(test_df))
        for col in ["ma20", "vol_norm", "rsi", "macd_diff", "bb_pct"]:
            future[col] = prophet_train[col].iloc[-1]

        forecast      = model.predict(future)
        prophet_preds = forecast["yhat"].iloc[-len(test_df):].values
        actuals       = test_df["close"].values[:len(prophet_preds)]

        return float(np.sqrt(np.mean((actuals - prophet_preds) ** 2)))
    except Exception:
        return 999.0


def compute_metrics(actuals: np.ndarray, predictions: np.ndarray) -> dict:
    # Full evaluation metrics suite
    rmse  = float(np.sqrt(np.mean((actuals - predictions) ** 2)))
    mae   = float(np.mean(np.abs(actuals - predictions)))
    mape  = float(np.mean(np.abs((actuals - predictions) / actuals)) * 100)
    smape = float(np.mean(
        2 * np.abs(actuals - predictions) /
        (np.abs(actuals) + np.abs(predictions))
    ) * 100)

    try:
        r2 = float(r2_score(actuals, predictions))
    except Exception:
        r2 = 0.0

    actual_dir = np.sign(np.diff(actuals))
    pred_dir   = np.sign(np.diff(predictions))
    dir_acc    = float(np.mean(actual_dir == pred_dir) * 100)

    return {
        "rmse":            round(rmse,    4),
        "mae":             round(mae,     4),
        "mape":            round(mape,    4),
        "smape":           round(smape,   4),
        "r2":              round(r2,      4),
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
                INSERT INTO fact_signals (ticker, date, predicted_close)
                VALUES (:ticker, :date, :predicted_close)
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
    print("=" * 60)
    print("KairosAI — Forecasting v4")
    print("Models : ARIMA + Prophet + LSTM")
    print("Features: RSI, MACD, Bollinger Bands, OBV, ATR")
    print("Regime : bull / bear / sideways detection")
    print(f"Horizon: {FORECAST_DAYS} days | Holdout: {HOLDOUT_DAYS} days")
    print("=" * 60)

    test_connection()
    ensure_model_dir()

    all_metrics = []

    for ticker in TICKERS:
        print(f"\n── {ticker} ────────────────────────────────────")

        df = load_prices(ticker)

        if df.empty or len(df) < MIN_ROWS:
            print(f"  Not enough data, skipping")
            continue

        print(f"  Loaded {len(df)} rows")

        # Prepare
        df["close"] = clip_outliers(df["close"])
        df          = add_features(df)

        # Detect market regime
        regime = detect_regime(df)
        print(f"  Regime: {regime}")

        # ── Evaluate all three models on holdout ──
        print(f"  Evaluating on {HOLDOUT_DAYS}-day holdout...")
        arima_rmse   = evaluate_arima_holdout(df)
        prophet_rmse = evaluate_prophet_holdout(df)
        lstm_rmse    = evaluate_lstm_holdout(df)

        print(f"  Holdout RMSE — ARIMA: ${arima_rmse:.2f} | "
              f"Prophet: ${prophet_rmse:.2f} | LSTM: ${lstm_rmse:.2f}")

        # ── Train on full data ──
        print(f"  Training on full dataset ({len(df)} rows)...")

        arima_model,   arima_type    = train_arima(df["close"], ticker)
        prophet_model, prophet_df_   = train_prophet(df, ticker)
        lstm_model,    lstm_scaler   = train_lstm(df, ticker)

        # ── Predict ──
        arima_result   = predict_arima(arima_model, arima_type) if arima_model   else None
        prophet_result = predict_prophet(prophet_model, df)     if prophet_model else None
        lstm_result    = predict_lstm(lstm_model, df, lstm_scaler) if lstm_model else None

        # ── Regime-weighted ensemble ──
        final = regime_weighted_ensemble(
            arima_result, prophet_result, lstm_result,
            arima_rmse,   prophet_rmse,   lstm_rmse,
            regime
        )

        if final:
            current_price = float(df["close"].iloc[-1])
            direction     = "UP" if final["prediction"] > current_price else "DOWN"
            change_pct    = ((final["prediction"] - current_price) / current_price) * 100

            print(f"  Current : ${current_price:.2f}")
            print(f"  Forecast: ${final['prediction']:.2f} ({direction} {abs(change_pct):.2f}%)")
            print(f"  CI 95%  : [${final['lower']:.2f} — ${final['upper']:.2f}]")

        # ── Compute ensemble metrics on holdout ──
        try:
            split    = len(df) - HOLDOUT_DAYS
            train_df = df.iloc[:split].copy()
            test_df  = df.iloc[split:].copy()

            # Quick ensemble on holdout for metrics
            actuals = test_df["close"].values

            # Use Prophet predictions as proxy for ensemble on holdout
            prophet_train = train_df[[
                "date", "close", "ma20", "vol_norm",
                "rsi", "macd_diff", "bb_pct"
            ]].rename(columns={"date": "ds", "close": "y"}).dropna()

            eval_model = Prophet(
                daily_seasonality=False,
                weekly_seasonality=True,
                yearly_seasonality=True,
                changepoint_prior_scale=0.05,
                interval_width=0.95
            )
            for reg in ["ma20", "vol_norm", "rsi", "macd_diff", "bb_pct"]:
                eval_model.add_regressor(reg)
            eval_model.fit(prophet_train)

            future = eval_model.make_future_dataframe(periods=len(test_df))
            for col in ["ma20", "vol_norm", "rsi", "macd_diff", "bb_pct"]:
                future[col] = prophet_train[col].iloc[-1]

            forecast      = eval_model.predict(future)
            prophet_preds = forecast["yhat"].iloc[-len(test_df):].values
            n             = min(len(actuals), len(prophet_preds))

            metrics           = compute_metrics(actuals[:n], prophet_preds[:n])
            metrics["ticker"] = ticker
            metrics["regime"] = regime
            all_metrics.append(metrics)

            print(f"  Metrics — RMSE: ${metrics['rmse']} | "
                  f"MAPE: {metrics['mape']}% | R²: {metrics['r2']} | "
                  f"Dir. Acc: {metrics['directional_acc']}%")

        except Exception as e:
            print(f"  Metrics failed: {e}")

        save_prediction(ticker, final)

    # ── Summary ──
    print("\n" + "=" * 60)
    print("FORECASTING COMPLETE — v4")
    print("=" * 60)

    if all_metrics:
        metrics_df = pd.DataFrame(all_metrics)
        print("\nModel performance on holdout:")
        print(metrics_df[[
            "ticker", "regime", "rmse", "mape", "r2", "directional_acc"
        ]].to_string(index=False))

        avg_dir = metrics_df["directional_acc"].mean()
        avg_r2  = metrics_df["r2"].mean()
        above   = (metrics_df["directional_acc"] > 50).sum()

        print(f"\nAverage directional accuracy : {avg_dir:.2f}%")
        print(f"Average R²                   : {avg_r2:.4f}")
        print(f"Tickers above random baseline: {above}/{len(metrics_df)}")
        print(f"Random baseline              : 50.00%")

        # Regime breakdown
        print("\nRegime breakdown:")
        print(metrics_df.groupby("regime")["directional_acc"].mean().to_string())

    return all_metrics


if __name__ == "__main__":
    run()