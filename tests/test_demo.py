from pathlib import Path
import sys

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code"))
from helper import evaluate, forecast, load_series, split_series


def test_holdout_and_metrics():
    series = load_series(ROOT / "examples" / "demand.csv")
    train, test = split_series(series)
    assert len(train) == 88 and len(test) == 8
    assert train.index.max() < test.index.min()
    assert evaluate([1, 2], [1, 4]) == {"mae": 1.0, "rmse": np.sqrt(2)}
    assert np.all(forecast(train, len(test), "baseline") == train.iloc[-1])


@pytest.mark.parametrize("method", ["arima", "sarimax"])
def test_model_forecasts(method):
    train, test = split_series(load_series(ROOT / "examples" / "demand.csv"))
    predicted = forecast(train, len(test), method)
    assert len(predicted) == len(test) and np.isfinite(predicted).all()
    assert all(np.isfinite(x) for x in evaluate(test, predicted).values())


def test_lstm_forecast():
    pytest.importorskip("tensorflow")
    train, test = split_series(load_series(ROOT / "examples" / "demand.csv"))
    predicted = forecast(train, len(test), "lstm", epochs=1)
    assert len(predicted) == len(test) and np.isfinite(predicted).all()


def test_invalid_holdout():
    with pytest.raises(ValueError):
        split_series(load_series(ROOT / "examples" / "demand.csv"), horizon=96)
