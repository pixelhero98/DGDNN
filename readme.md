# DGDNN

Dynamic Graph Diffusion Neural Network for stock movement prediction.

This repository can now build graph datasets from freshly fetched market data.
By default, the training CLI fetches the current S&P 500 constituents from
Wikipedia at runtime, normalizes the symbols for Yahoo Finance, downloads OHLCV
data with `yfinance`, and trains on all current S&P 500 tickers unless a ticker
subset is provided.

## Installation

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

Install the PyTorch build appropriate for your hardware if the default `torch`
package is not the one you need.

## Quickstart

Run a small S&P 500 subset:

```powershell
.\.venv\Scripts\python.exe -m Train_Eval.next_day_movement_prediction `
  --tickers AAPL,MSFT,NVDA,AMZN,GOOGL `
  --start 2023-01-01 `
  --end 2024-12-31 `
  --epochs 1
```

Run the default current S&P 500 universe by omitting `--tickers`:

```powershell
.\.venv\Scripts\python.exe -m Train_Eval.next_day_movement_prediction `
  --start 2023-01-01 `
  --end 2024-12-31
```

Downloaded CSVs are written to `data/raw/`; serialized graph tensors are written
to `data/graphs/`. These generated artifacts are ignored by git.

## Data Behavior

- Default market: current S&P 500 constituents from Wikipedia.
- Default ticker set: all current S&P 500 constituents in table order.
- Requested tickers: validated against the current S&P 500 table.
- Yahoo symbols: normalized automatically, for example `BRK.B` becomes `BRK-B`.
- Missing CSVs: downloaded automatically unless `--no-download` is set.
- Unavailable Yahoo data: reported clearly; use `--drop-failed-downloads` to
  continue after failed downloads.

The old Google Drive and Dropbox data links may still be useful for historical
reproduction, but they are no longer required for a fresh run.

## Features And Graphs

The default feature transform is ratio return:

```text
(x_t - x_{t-1}) / max(abs(x_{t-1}), eps)
```

Use `--feature-transform log` to recover the previous `log1p(raw + eps)` path.
The default adjacency uses continuous kNN mutual information. The optional
positive-only post-log adjacency filter is off by default; enable it with
`--positive-adj-filter`.

Ratio features require one common trading date before the experiment start.
The CLI downloads a small lookback buffer automatically.

## Caveats

Yahoo Finance availability can change over time and may be rate-limited. Current
S&P 500 constituents introduce survivorship bias for historical backtests because
the ticker universe is selected at run time, not as-of the historical date.

Serialized graph `.pt` files are trusted local artifacts produced by this repo.
Do not load graph caches from untrusted sources.
