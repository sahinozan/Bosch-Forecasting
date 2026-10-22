"""Evaluate a small forecasting model on a synthetic holdout period."""
import argparse
import json
from pathlib import Path

import pandas as pd

from helper import evaluate, forecast, load_series, split_series

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method", choices=["baseline", "arima", "sarimax", "lstm"], default="arima")
    parser.add_argument("--item", default="SYN-ITEM-01")
    parser.add_argument("--horizon", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=5)
    args = parser.parse_args()
    train, test = split_series(load_series(ROOT / "examples" / "demand.csv", args.item), args.horizon)
    predicted = forecast(train, len(test), args.method, epochs=args.epochs)
    metrics = evaluate(test, predicted)
    output = ROOT / "outputs"
    output.mkdir(exist_ok=True)
    pd.DataFrame({"actual": test, "predicted": predicted}).to_csv(output / f"{args.method}_forecast.csv")
    (output / f"{args.method}_metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    print(json.dumps({"synthetic_demo": True, "method": args.method, **metrics}))


if __name__ == "__main__":
    main()
