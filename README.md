# Bosch Senior Project — Forecasting

This repository contains the forecasting work from a bachelor's senior project completed as a team in collaboration with Bosch. We explored patterns in component demand and experimented with time-series models to support production and inventory planning.

The original analysis depended on company-specific datasets and exports. This repository is shared as an academic portfolio project; the current public examples demonstrate the approaches using fictional demand.

## What the project covered

- Preparing time-series data and exploring trends and seasonality.
- Experimenting with ARIMA, seasonal ARIMA/SARIMAX, and LSTM models.
- Comparing forecasts with observed values using error metrics.

**Technologies:** Python, pandas, NumPy, statsmodels, TensorFlow/Keras, and Jupyter notebooks.

## Sanitization for public sharing

Original datasets, material identifiers, fitted parameter tables, saved notebook outputs, and personal paths have been removed from this version. The company-specific data parser has been replaced with a simple CSV example, and the models use illustrative settings.

The sample demand is generated from scratch by `generate_demo.py`, without using original records or their statistics. Its trends and seasonal patterns are invented. Results describe only the synthetic examples and do not reproduce the original experiments or demonstrate performance on company data.

## Try the example

With Python 3.11 or newer, run these commands from the repository directory, preferably in a virtual environment:

```sh
python -m pip install -r requirements.txt
python code/demo.py --method arima
```

The demo holds out the final eight weeks and writes forecasts and metrics to `outputs/`. Other options are `--method sarimax` and `--method baseline`.

For the optional LSTM example:

```sh
python -m pip install -r requirements-lstm.txt
python code/demo.py --method lstm --epochs 5
```

The `analysis/` and `forecasting/` folders contain notebooks for exploring the examples. Regenerate the sample data with `python generate_demo.py`.

## Related senior-project repositories

- [Excel Integration](https://github.com/sahinozan/Bosch-Excel-Integration) — production-plan processing and Excel reporting.
- [Reinforcement Learning](https://github.com/sahinozan/Bosch-Reinforcement-Learning) — a custom inventory environment and reinforcement learning experiments.
