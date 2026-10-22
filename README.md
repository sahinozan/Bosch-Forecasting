# Bosch-Forecasting

A public time-series forecasting demonstration using independently generated fictional demand. The repository name remains a historical project label. No committed sample represents company demand, material identifiers, production volumes, or fitted business parameters.

**Sanitized public demo:** This project has been adapted for public sharing through synthetic-data replacement and generalization of company-specific behavior. [Read what changed and why](#sanitization-for-public-sharing).

## Run

Use Python 3.11 or newer in a virtual environment:

```sh
python -m pip install -r requirements.txt
python code/demo.py --method arima
python code/demo.py --method sarimax
python code/demo.py --method baseline
```

The demo holds out the final eight weeks, fits only on earlier observations, and writes forecasts and metrics to the ignored `outputs/` directory. These are illustrative results, not claims about real-world predictive performance.

For the optional LSTM example:

```sh
python -m pip install -r requirements-lstm.txt
python code/demo.py --method lstm --epochs 5
```

LSTM uses lagged training windows and recursive prediction. It does not use holdout values as model inputs. The small model and short training run prioritize an easy demonstration over accuracy.

## Examples and notebooks

`examples/demand.csv` contains four invented `SYN-ITEM-*` series over 96 arbitrary weeks. Recreate it with `python generate_demo.py`. The generator uses a fixed seed, a simple sine wave, a small trend, and random noise. It does not read or preserve statistics from any original dataset.

The notebooks in `analysis/` and `forecasting/` contain only source cells; saved outputs and machine metadata are omitted. Install Jupyter to open them. The parser and export examples use the fictional CSV. The forecasting notebooks require the corresponding dependencies above.

## Sanitization for public sharing

This edition demonstrates the project's technical ideas while excluding operational records and personal information. The sample data is generated from scratch by `generate_demo.py`, without reading original records or deriving values from their statistics. Identifiers, quantities, dates, trends, and seasonal patterns in the examples are invented.

Original spreadsheets, material tables, fitted parameter dictionaries, notebook outputs, and personal filesystem paths have been removed. The original corporate spreadsheet parser has been replaced with a documented `date,item_id,quantity` CSV schema. ARIMA, seasonal ARIMA, and LSTM remain as generic examples, with newly chosen illustrative settings.

Identifying document and notebook metadata have also been removed from the published files. The public demo starts with a fresh initial commit containing the sanitized edition. These changes are described as **sanitization and synthetic-data replacement**. New forecasts and metrics describe only the fictional examples and do not reproduce the original experiments or establish performance on operational data.

Run the checks with `python -m pip install pytest` and `python -m pytest tests`. The optional LSTM check skips if TensorFlow is unavailable.
