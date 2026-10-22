"""Forecasting utilities for the synthetic public demo edition.

The organization-specific export parser and fitted parameter tables have been
replaced. Models are trained only on the supplied training series.
"""
import numpy as np
import pandas as pd


def load_series(path, item="SYN-ITEM-01"):
    frame = pd.read_csv(path, parse_dates=["date"])
    if not {"date", "item_id", "quantity"}.issubset(frame):
        raise ValueError("Expected date, item_id, quantity columns")
    rows = frame.loc[frame["item_id"] == item].sort_values("date")
    if rows.empty or rows["date"].duplicated().any() or rows.isna().any().any():
        raise ValueError("Expected one nonempty weekly series without missing or duplicate dates")
    series = rows.set_index("date")["quantity"].astype(float)
    if not np.isfinite(series).all() or (series < 0).any():
        raise ValueError("Quantities must be finite and nonnegative")
    series = series.asfreq("W-MON")
    if series.isna().any():
        raise ValueError("Dates must be consecutive Mondays")
    return series


def split_series(series, horizon=8):
    if not 0 < horizon < len(series):
        raise ValueError("horizon must leave both training and holdout observations")
    return series.iloc[:-horizon].copy(), series.iloc[-horizon:].copy()


def forecast(train, horizon=8, method="arima", epochs=5):
    values = np.asarray(train, dtype=float)
    if values.ndim != 1 or not len(values) or not np.isfinite(values).all():
        raise ValueError("Expected a finite, nonempty training series")
    if horizon < 1:
        raise ValueError("horizon must be positive")
    if method == "baseline":
        return np.repeat(values[-1], horizon)
    if method == "arima":
        from statsmodels.tsa.arima.model import ARIMA
        model = ARIMA(values, order=(1, 1, 0)).fit()
        return np.asarray(model.forecast(steps=horizon))
    if method == "sarimax":
        from statsmodels.tsa.statespace.sarimax import SARIMAX
        # Illustrative settings, unrelated to any original fitted parameters.
        model = SARIMAX(values, order=(1, 0, 0), seasonal_order=(1, 0, 0, 12)).fit(disp=False)
        return np.asarray(model.forecast(steps=horizon))
    if method == "lstm":
        return forecast_lstm(values, horizon, epochs=epochs)
    raise ValueError(f"Unknown method: {method}")


def forecast_lstm(values, horizon, epochs=5, lookback=12):
    """Fit on lagged training windows; recursively forecast without holdout input."""
    if len(values) <= lookback or epochs < 1:
        raise ValueError("Need more than lookback observations and a positive epoch count")
    import tensorflow as tf

    tf.keras.utils.set_random_seed(42)
    scale = max(float(np.max(np.abs(values))), 1.0)
    normalized = values / scale
    x = np.array([normalized[i - lookback:i] for i in range(lookback, len(values))])
    y = normalized[lookback:]
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(lookback, 1)),
        tf.keras.layers.LSTM(8),
        tf.keras.layers.Dense(1),
    ])
    model.compile(optimizer="adam", loss="mse")
    model.fit(x[..., None], y, epochs=epochs, batch_size=16, shuffle=False, verbose=0)
    window = normalized[-lookback:].copy()
    predictions = []
    for _ in range(horizon):
        value = float(model(window.reshape(1, lookback, 1), training=False).numpy()[0, 0])
        predictions.append(value * scale)
        window = np.append(window[1:], value)
    return np.asarray(predictions)


def evaluate(actual, predicted):
    actual, predicted = np.asarray(actual, dtype=float), np.asarray(predicted, dtype=float)
    if actual.shape != predicted.shape or not actual.size:
        raise ValueError("Actual and predicted values must have the same nonempty shape")
    if not np.isfinite(actual).all() or not np.isfinite(predicted).all():
        raise ValueError("Metrics require finite values")
    error = actual - predicted
    return {"mae": float(np.mean(np.abs(error))), "rmse": float(np.sqrt(np.mean(error ** 2)))}
