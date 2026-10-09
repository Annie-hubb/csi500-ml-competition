# CSI 500 stock-selection experiments

A course competition project comparing an XGBoost baseline, a five-seed ensemble, and rule-based stock-selection strategies over five-trading-day holding periods. This public snapshot contains selected student-authored experiment scripts and aggregate backtest outputs. It does **not** include raw market data, submitted portfolios, course starter code, or the course Git history.

## Data and features

The local research snapshot contained **282,874 daily price observations** and **499 CSI 500 constituents** (approximately 283K observations and 500 stocks). The raw data is intentionally excluded. `xgboost_v3_robust.py` defines 25 price, momentum, volatility, liquidity, RSI, and Amihud features, then converts them to within-date cross-sectional ranks. It also defines six trading-risk flags. The portfolio constructor applies long-only, minimum-name, maximum-weight, suspension/limit, and liquidity rules.

The model target ranks forward excess returns relative to the CSI 500 index. Model training uses a specified as-of date, only labels whose exit date is already known, and a time-based embargo before the validation period. Walk-forward folds use a five-trading-day horizon.

## Saved comparison

`results/four_versions_h5_wide.csv` contains 42 historical evaluation windows, from 2025-07-01 through 2026-05-08. `results/four_versions_h5_summary.csv` reports arithmetic **mean excess return per five-day window** relative to CSI 500; these are not annualized returns.

| Strategy | Mean excess per window | Winning windows |
| --- | ---: | ---: |
| Course XGBoost baseline | -0.1055% | 40.48% |
| V2 five-seed XGBoost ensemble | +0.1594% | 59.52% |
| V3 mean-reversion/liquidity rule | +0.2218% | 54.76% |
| V3 liquidity-ranked long-only top 50 | +0.7326% | 61.90% |

The highest reported result belongs to a **rule-based liquidity strategy**, not to XGBoost. The strategy was selected after comparing historical windows, so the table should not be presented as an untouched final holdout or evidence of live trading performance. Transaction costs, slippage, changing constituent membership, and market-regime robustness are not established by these files.

## Reproducing the experiment

Python 3.10+ is recommended. Install dependencies with `pip install -r requirements.txt`. Obtain market data independently and follow the provider's license and usage terms. The original data acquisition used [AkShare](https://github.com/akfamily/akshare) functions for CSI 500 constituents (`index_stock_cons_csindex`, symbol `000905`), forward-adjusted stock OHLCV (`stock_zh_a_daily`, `adjust="qfq"`), and CSI 500 index history (`stock_zh_index_daily`). Use only information available by each simulated decision date.

Create `data/prices.parquet` with one row per stock/date and columns `date`, `stock_code`, `open`, `close`, `high`, `low`, `volume`, `amount`, `turnover`, `pct_change`; create `data/index.parquet` with `date` and index `close`. The local snapshot's `data/constituents.csv` was the current universe at collection time, so it is not a point-in-time membership archive. No data file is committed here.

The self-contained V3 feature and portfolio script can be run with your own compatible data:

```bash
python xgboost_v3_robust.py --as-of YYYYMMDD --horizon 5 --alpha-mode liquidity --top-k 50 --out submission.csv
python walkforward_validate.py --start YYYYMMDD --end YYYYMMDD --horizon 5 --step 5 --alpha-mode liquidity --out results/local_folds.csv
```

`xgboost_v2.py` and `compare_walkforward_h5.py` are included to document the original comparison, but they import course-provided `features.py` and `baseline_xgboost.py`, which are not redistributed. To rerun the four-way comparison, supply those course starter modules privately, along with a compatible data snapshot. Recreated results can differ when data vendors revise prices or membership.

## Provenance

The repository is a selective public presentation of work from an individual 2026 course competition. The aggregate CSVs are the saved outputs of `compare_walkforward_h5.py`. Code comments and saved results are evidence of the implementation and reported experiment; they do not by themselves prove future performance. Some coding work used AI assistance, while experiment design and interpretation remain the author's responsibility.
