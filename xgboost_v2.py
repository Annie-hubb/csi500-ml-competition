"""
xgboost_v2.py  —  Improved CSI500 stock-selection pipeline
==========================================================

Drop-in replacement for baseline_xgboost.py.  Same CLI:

    python xgboost_v2.py --out submissions/week1.csv

Changes vs the original baseline (in priority order):

  1. use_60d=True for features
     Your own features.py comment notes that ret_60d is the single
     highest-IC factor in A-shares (~0.04–0.07 standalone).  The
     baseline left it off.  Turning it on gives the model the
     long-horizon momentum signal it was missing.

  2. Risk filter at prediction time
     The competition rules say a suspended stock contributes 0
     (effectively cash drag), and a stock that hit limit-up yesterday
     gaps down on average.  We drop these BEFORE picking top-K, plus
     bottom-decile by 20d turnover (illiquid → noisy realised return).

  3. 5-seed ensemble
     XGBoost on a 1-year panel has high seed variance.  Averaging
     5 seeds typically lifts IC by 0.01–0.02 and roughly halves
     the std across val windows.

  4. Target winsorization (1% / 99% per date)
     5-day forward returns have fat tails.  Clipping the per-date
     extremes stops a handful of huge moves from dominating the
     gradient and producing unstable trees.

  5. top_k = 40 (was 50)
     With rank-weighting + 10% cap, names ranked 41-50 carry ~5% of
     total weight combined.  Cutting the long tail concentrates more
     on high-confidence picks without violating the >=30 constraint.

Hard constraints honoured (rules §3): >=30 names, max weight 0.10,
weights non-negative summing to 1.0 within 1e-4.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from scipy.stats import spearmanr

from features import (
    build_features,
    training_frame,
    prediction_frame,
    get_feature_cols,
    TARGET_COLUMN,
    FORWARD_HORIZON,
)

# ─────────────────────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────────────────────
DATA_DIR        = Path(__file__).parent / "data"

USE_60D         = True       # turn on ret_60d / close_over_ma60 / ret_60d_rank
VAL_DAYS        = 10         # validation window (last N trading days of train pool)
EMBARGO_DAYS    = 6          # > FORWARD_HORIZON to prevent label leakage
N_SEEDS         = 5          # ensemble size
TARGET_WINSOR   = 0.01       # winsorize top/bottom 1% of target per date

DEFAULT_TOP_K   = 40         # baseline used 50; 40 is more concentrated
MIN_STOCKS      = 30         # rule §3.2
MAX_WEIGHT      = 0.10       # rule §3.3
LIQUIDITY_PCT   = 0.10       # drop bottom 10% by 20d-avg turnover


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def winsorize_per_date(panel: pd.DataFrame, col: str, pct: float = 0.01) -> pd.DataFrame:
    """Clip top/bottom `pct` of `col` within each date.  Modifies in place."""
    def _w(s: pd.Series) -> pd.Series:
        lo, hi = s.quantile(pct), s.quantile(1 - pct)
        return s.clip(lo, hi)
    panel[col] = panel.groupby("date")[col].transform(_w)
    return panel


def rank_ic(y_true: np.ndarray, y_pred: np.ndarray, dates: np.ndarray) -> float:
    """Daily cross-sectional Spearman correlation, averaged."""
    ics: list[float] = []
    for d in np.unique(dates):
        m = dates == d
        if m.sum() < 20:
            continue
        rho, _ = spearmanr(y_true[m], y_pred[m])
        if not np.isnan(rho):
            ics.append(float(rho))
    return float(np.mean(ics)) if ics else float("nan")


def train_one(train_df: pd.DataFrame, val_df: pd.DataFrame,
              feature_cols: list[str], seed: int) -> xgb.XGBRegressor:
    model = xgb.XGBRegressor(
        n_estimators=500,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=10,
        reg_lambda=1.0,
        tree_method="hist",
        n_jobs=-1,
        early_stopping_rounds=30,
        random_state=seed,
    )
    model.fit(
        train_df[feature_cols], train_df[TARGET_COLUMN],
        eval_set=[(val_df[feature_cols], val_df[TARGET_COLUMN])],
        verbose=False,
    )
    return model


def build_portfolio(scores: pd.Series, eligible: pd.Series,
                    top_k: int = DEFAULT_TOP_K) -> pd.Series:
    """
    Top-K rank-weighted portfolio with 10% cap and spillover redistribution.

    Parameters
    ----------
    scores   : predicted score, indexed by stock_code (full universe)
    eligible : bool mask, indexed by stock_code (True = ok to hold)
    top_k    : number of names to select
    """
    if top_k < MIN_STOCKS:
        raise ValueError(f"top_k must be >= {MIN_STOCKS}")

    pool = scores[eligible]
    if len(pool) < top_k:
        # Edge case: filter ate too many names.  Fall back to full universe
        # so we never violate the >=30 rule.
        print(f"   ! only {len(pool)} eligible after filter; falling back to full universe")
        pool = scores

    chosen = pool.sort_values(ascending=False).head(top_k).copy()

    # Linear rank weights: best stock gets rank=top_k, ..., worst gets 1
    ranks = np.arange(top_k, 0, -1, dtype=float)
    w = pd.Series(ranks / ranks.sum(), index=chosen.index)

    # Iteratively cap at MAX_WEIGHT, redistribute excess pro-rata to uncapped
    for _ in range(50):
        over = w > MAX_WEIGHT
        if not over.any():
            break
        excess = (w[over] - MAX_WEIGHT).sum()
        w[over] = MAX_WEIGHT
        free = ~over
        if not free.any():
            break
        w[free] += excess * w[free] / w[free].sum()

    # Sanity asserts (also enforced by validate_submission.py server-side)
    assert abs(w.sum() - 1.0) < 1e-6, f"weights sum to {w.sum()}"
    assert (w <= MAX_WEIGHT + 1e-9).all(), "cap violated"
    assert (w > 0).sum() >= MIN_STOCKS, "too few names"
    return w


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--prices",  default=str(DATA_DIR / "prices.parquet"))
    p.add_argument("--as-of",   default=None,
                   help="YYYYMMDD; defaults to latest date in data")
    p.add_argument("--top-k",   type=int, default=DEFAULT_TOP_K)
    p.add_argument("--n-seeds", type=int, default=N_SEEDS)
    p.add_argument("--out",     default="submission.csv")
    args = p.parse_args()

    # ── Load ──────────────────────────────────────────────────────────────
    print(f">> Loading {args.prices}")
    prices = pd.read_parquet(args.prices)
    print(f"   {len(prices):,} rows, {prices['stock_code'].nunique()} stocks, "
          f"{prices['date'].min().date()} → {prices['date'].max().date()}")

    # ── Features ──────────────────────────────────────────────────────────
    print(f">> Building features (use_60d={USE_60D})")
    panel = build_features(prices, use_60d=USE_60D)
    feature_cols = [c for c in get_feature_cols(use_60d=USE_60D) if c in panel.columns]
    print(f"   {len(feature_cols)} features: {feature_cols}")

    # Target winsorization
    print(f">> Winsorizing target at {TARGET_WINSOR} / {1 - TARGET_WINSOR} per date")
    panel = winsorize_per_date(panel, TARGET_COLUMN, pct=TARGET_WINSOR)

    # ── Train / val split ─────────────────────────────────────────────────
    # Cap training data so labels never reach into the prediction date when
    # running with --as-of for backtesting.
    as_of_ts      = pd.Timestamp(args.as_of) if args.as_of else panel["date"].max()
    trading_dates = np.sort(panel["date"].unique())
    as_of_idx     = int(np.searchsorted(trading_dates, np.datetime64(as_of_ts)))
    cutoff_idx    = max(0, as_of_idx - FORWARD_HORIZON)
    train_cutoff  = pd.Timestamp(trading_dates[cutoff_idx])
    train_pool    = training_frame(panel, max_date=train_cutoff, use_60d=USE_60D)

    all_dates = np.sort(train_pool["date"].unique())
    if len(all_dates) < VAL_DAYS + EMBARGO_DAYS + 20:
        raise RuntimeError(f"only {len(all_dates)} training dates available "
                           f"(need >= {VAL_DAYS + EMBARGO_DAYS + 20})")

    val_start = pd.Timestamp(all_dates[-VAL_DAYS])
    train_end = pd.Timestamp(all_dates[-(VAL_DAYS + EMBARGO_DAYS + 1)])
    train_df  = train_pool[train_pool["date"] <= train_end]
    val_df    = train_pool[train_pool["date"] >= val_start]
    print(f"   train: {len(train_df):,} rows ≤ {train_end.date()}")
    print(f"   embargo: {EMBARGO_DAYS} days (discarded)")
    print(f"   val:   {len(val_df):,} rows ≥ {val_start.date()}")

    # ── Ensemble training ─────────────────────────────────────────────────
    print(f">> Training {args.n_seeds}-seed XGBoost ensemble")
    models: list[xgb.XGBRegressor] = []
    val_preds: list[np.ndarray] = []
    y_val   = val_df[TARGET_COLUMN].to_numpy()
    d_val   = val_df["date"].to_numpy()
    for seed in range(args.n_seeds):
        m = train_one(train_df, val_df, feature_cols, seed=seed)
        vp = m.predict(val_df[feature_cols])
        models.append(m)
        val_preds.append(vp)
        print(f"   seed {seed}: IC = {rank_ic(y_val, vp, d_val):+.4f}")

    val_pred_avg = np.mean(val_preds, axis=0)
    print(f"   ── ensemble IC: {rank_ic(y_val, val_pred_avg, d_val):+.4f} ──")

    # ── Predict on as-of date ─────────────────────────────────────────────
    print(">> Generating prediction-day scores")
    pred_df = prediction_frame(panel, as_of=args.as_of, use_60d=USE_60D)
    if pred_df.empty:
        raise RuntimeError(f"empty prediction frame for as_of={args.as_of}")
    pred_date = pred_df["date"].iloc[0]
    print(f"   prediction date: {pred_date.date()}, candidates: {len(pred_df)}")

    pred_X = pred_df[feature_cols]
    preds  = np.mean([m.predict(pred_X) for m in models], axis=0)
    pred_df = pred_df.assign(score=preds)
    indexed = pred_df.set_index("stock_code")

    # ── Eligibility / risk filter ─────────────────────────────────────────
    print(">> Applying risk filter")
    eligible = pd.Series(True, index=indexed.index)
    n_total = len(eligible)

    if "suspend_flag" in indexed.columns:
        flag = indexed["suspend_flag"] >= 1
        eligible &= ~flag
        print(f"   - suspended yesterday   : drop {int(flag.sum())}")

    if "volume" in indexed.columns:
        # Catches "suspended TODAY" — yesterday's flag misses it
        no_trade = indexed["volume"] == 0
        eligible &= ~no_trade
        print(f"   - volume == 0 today     : drop {int(no_trade.sum())}")

    if "limit_up_flag" in indexed.columns:
        flag = indexed["limit_up_flag"] >= 1
        eligible &= ~flag
        print(f"   - limit-up yesterday    : drop {int(flag.sum())}")

    if "turnover_ma_20d" in indexed.columns:
        thr = indexed["turnover_ma_20d"].quantile(LIQUIDITY_PCT)
        illiquid = indexed["turnover_ma_20d"] < thr
        eligible &= ~illiquid
        print(f"   - bottom {int(LIQUIDITY_PCT*100)}% turnover     : drop {int(illiquid.sum())}")

    print(f"   eligible: {int(eligible.sum())} / {n_total}")

    # ── Portfolio ─────────────────────────────────────────────────────────
    scores  = indexed["score"]
    weights = build_portfolio(scores, eligible, top_k=args.top_k)

    # Sanity print: top 10 holdings
    top10 = weights.sort_values(ascending=False).head(10)
    print("\n   Top 10 holdings:")
    for code, w in top10.items():
        print(f"     {code}  weight={w:.4f}  score={scores.loc[code]:+.4f}")

    # ── Write CSV ─────────────────────────────────────────────────────────
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out = pd.DataFrame({"stock_code": weights.index, "weight": weights.values})
    out.to_csv(out_path, index=False)
    print(f"\n>> Wrote {len(out)} names → {out_path}")
    print(f"   weights: min={out['weight'].min():.4f}  "
          f"max={out['weight'].max():.4f}  sum={out['weight'].sum():.6f}")
    print(f"\n   Run `python validate_submission.py {out_path}` before uploading.")


if __name__ == "__main__":
    main()
