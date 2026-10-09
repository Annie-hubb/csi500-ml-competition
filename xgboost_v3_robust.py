"""
xgboost_v3_robust.py -- robust short-horizon CSI500 stock-selection pipeline.

This file is "V3": one script, multiple `--alpha-mode` strategies.
`liquidity` / `ma-revert-liq` / `hybrid` / `model` are all V3 modes — not separate
projects. `liquidity` is rule-only (no XGB); `model` / `hybrid` train trees on
ranked excess-return targets.

Design goals:
  * force an explicit as-of date so a refreshed parquet cannot create hidden
    lookahead;
  * train only on labels whose exit date is known by that as-of date;
  * use short-horizon reversal / overextension / liquidity features;
  * predict a cross-sectional target, not raw returns dominated by market beta;
  * build a diversified long-only portfolio with a modest score tilt.

Example for submission 2 (walk-forward winner in our tests: liquidity, top_k=50):

    python xgboost_v3_robust.py --as-of 20260508 --horizon 5 \
        --alpha-mode liquidity --top-k 50 --out submissions/week2_v3.csv

For XGBoost scoring, use `--alpha-mode hybrid` or `--alpha-mode model`.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from scipy.stats import spearmanr


DATA_DIR = Path(__file__).parent / "data"

MIN_STOCKS = 30
MAX_WEIGHT = 0.10

DEFAULT_TOP_K = 50
DEFAULT_N_SEEDS = 3
DEFAULT_VAL_DAYS = 40
DEFAULT_ALPHA_MODE = "liquidity"

TARGET_COLUMN = "target_cs_rank"

RAW_FEATURES = [
    "ret_1d",
    "ret_3d",
    "ret_5d",
    "ret_10d",
    "ret_20d",
    "ret_60d",
    "close_over_ma5",
    "close_over_ma10",
    "close_over_ma20",
    "close_over_ma60",
    "vol_5d",
    "vol_10d",
    "vol_20d",
    "volume_z_5d",
    "volume_z_20d",
    "turnover_ma_20d",
    "turnover_z_20d",
    "intraday_ret",
    "gap_ret",
    "daily_range",
    "close_location",
    "amihud_20d",
    "drawdown_20d",
    "rsi_6",
    "rsi_14",
]

RANK_FEATURES = [f"{c}_rank" for c in RAW_FEATURES]

RISK_COLUMNS = [
    "suspend_flag",
    "no_trade_today",
    "limit_up_yday",
    "limit_dn_yday",
    "limit_up_today",
    "limit_dn_today",
]


@dataclass(frozen=True)
class PortfolioConfig:
    top_k: int = DEFAULT_TOP_K
    equal_weight_share: float = 0.70
    max_weight: float = MAX_WEIGHT
    liquidity_floor_pct: float = 0.10
    model_weight: float = 0.60
    alpha_mode: str = DEFAULT_ALPHA_MODE


def _safe_div(num: pd.Series, den: pd.Series) -> pd.Series:
    return num / den.replace(0, np.nan)


def _rsi(close: pd.Series, period: int) -> pd.Series:
    delta = close.diff()
    up = delta.clip(lower=0).rolling(period, min_periods=max(3, period // 2)).mean()
    dn = (-delta.clip(upper=0)).rolling(period, min_periods=max(3, period // 2)).mean()
    rs = up / dn.replace(0, np.nan)
    return 100 - 100 / (1 + rs)


def _per_stock_features(df: pd.DataFrame, horizon: int) -> pd.DataFrame:
    df = df.sort_values("date").copy()

    close = df["close"].astype(float)
    open_ = df["open"].astype(float) if "open" in df else close
    high = df["high"].astype(float) if "high" in df else close
    low = df["low"].astype(float) if "low" in df else close
    volume = df["volume"].astype(float)
    turnover = df["turnover"].astype(float) if "turnover" in df else pd.Series(np.nan, index=df.index)
    amount = df["amount"].astype(float) if "amount" in df else close * volume

    r1 = close.pct_change(1)
    for h in [1, 3, 5, 10, 20, 60]:
        df[f"ret_{h}d"] = close.pct_change(h)

    for w in [5, 10, 20, 60]:
        minp = max(3, w // 2)
        df[f"close_over_ma{w}"] = close / close.rolling(w, min_periods=minp).mean() - 1.0

    for w in [5, 10, 20]:
        minp = max(3, w // 2)
        df[f"vol_{w}d"] = r1.rolling(w, min_periods=minp).std()

    for w in [5, 20]:
        minp = max(3, w // 2)
        vm = volume.rolling(w, min_periods=minp).mean()
        vs = volume.rolling(w, min_periods=minp).std()
        df[f"volume_z_{w}d"] = (volume - vm) / vs.replace(0, np.nan)

    tm = turnover.rolling(20, min_periods=10).mean()
    ts = turnover.rolling(20, min_periods=10).std()
    df["turnover_ma_20d"] = tm
    df["turnover_z_20d"] = (turnover - tm) / ts.replace(0, np.nan)

    prev_close = close.shift(1)
    df["intraday_ret"] = _safe_div(close, open_) - 1.0
    df["gap_ret"] = _safe_div(open_, prev_close) - 1.0
    df["daily_range"] = _safe_div(high - low, prev_close)
    df["close_location"] = _safe_div(close - low, high - low) - 0.5
    df["amihud_20d"] = (r1.abs() / amount.replace(0, np.nan)).rolling(20, min_periods=10).mean()
    df["drawdown_20d"] = close / close.rolling(20, min_periods=10).max() - 1.0
    df["rsi_6"] = _rsi(close, 6)
    df["rsi_14"] = _rsi(close, 14)

    pct = df["pct_change"].astype(float) if "pct_change" in df else r1 * 100.0
    df["suspend_flag"] = (volume.shift(1) == 0).astype(float)
    df["no_trade_today"] = (volume == 0).astype(float)
    df["limit_up_yday"] = (pct.shift(1) >= 9.8).astype(float)
    df["limit_dn_yday"] = (pct.shift(1) <= -9.8).astype(float)
    df["limit_up_today"] = (pct >= 9.8).astype(float)
    df["limit_dn_today"] = (pct <= -9.8).astype(float)

    df[f"target_{horizon}d_raw"] = close.shift(-horizon) / close - 1.0
    return df


def _add_cross_sectional_features(panel: pd.DataFrame) -> pd.DataFrame:
    panel = panel.copy()
    for col in RAW_FEATURES:
        if col in panel.columns:
            panel[f"{col}_rank"] = panel.groupby("date")[col].rank(pct=True, method="average") - 0.5
    return panel


def _fill_feature_nans(panel: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    panel = panel.copy()
    for col in cols:
        if col not in panel.columns:
            continue
        mask = panel[col].isna()
        if mask.any():
            med = panel.groupby("date")[col].transform("median")
            panel.loc[mask, col] = med[mask]
        panel[col] = panel[col].fillna(0.0)
    for col in RISK_COLUMNS:
        if col in panel.columns:
            panel[col] = panel[col].fillna(0.0)
    return panel


def build_feature_panel(prices: pd.DataFrame, index_df: pd.DataFrame, horizon: int) -> pd.DataFrame:
    required = {"date", "stock_code", "open", "close", "high", "low", "volume"}
    missing = required - set(prices.columns)
    if missing:
        raise ValueError(f"prices missing required columns: {sorted(missing)}")

    prices = prices.copy()
    prices["date"] = pd.to_datetime(prices["date"])
    index_df = index_df.copy()
    index_df["date"] = pd.to_datetime(index_df["date"])

    panel = pd.concat(
        [_per_stock_features(g, horizon=horizon) for _, g in prices.groupby("stock_code", sort=False)],
        ignore_index=True,
    )

    index_df = index_df.sort_values("date").copy()
    index_df[f"benchmark_{horizon}d"] = (
        index_df["close"].shift(-horizon) / index_df["close"] - 1.0
    )
    panel = panel.merge(index_df[["date", f"benchmark_{horizon}d"]], on="date", how="left")
    panel[f"target_{horizon}d_excess"] = (
        panel[f"target_{horizon}d_raw"] - panel[f"benchmark_{horizon}d"]
    )

    panel = _add_cross_sectional_features(panel)
    panel[TARGET_COLUMN] = (
        panel.groupby("date")[f"target_{horizon}d_excess"]
        .rank(pct=True, method="average") - 0.5
    )

    panel = _fill_feature_nans(panel, RANK_FEATURES)
    return panel


def get_feature_cols() -> list[str]:
    return RANK_FEATURES + RISK_COLUMNS


def resolve_asof(trading_dates: np.ndarray, as_of: str | pd.Timestamp) -> pd.Timestamp:
    as_of_ts = pd.Timestamp(as_of)
    idx = int(np.searchsorted(trading_dates, np.datetime64(as_of_ts), side="right") - 1)
    if idx < 0:
        raise ValueError(f"as_of={as_of_ts.date()} is before first trading date")
    resolved = pd.Timestamp(trading_dates[idx])
    if resolved != as_of_ts:
        print(f"   as_of {as_of_ts.date()} is not a trading day; using previous trading day {resolved.date()}")
    return resolved


def known_label_cutoff(trading_dates: np.ndarray, as_of: pd.Timestamp, horizon: int) -> pd.Timestamp:
    idx = int(np.searchsorted(trading_dates, np.datetime64(as_of)))
    if idx < horizon:
        raise ValueError("not enough history before as_of to construct known-label cutoff")
    return pd.Timestamp(trading_dates[idx - horizon])


def make_train_val(
    panel: pd.DataFrame,
    as_of: pd.Timestamp,
    horizon: int,
    val_days: int = DEFAULT_VAL_DAYS,
    embargo_days: int | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Timestamp, pd.Timestamp, pd.Timestamp]:
    feature_cols = get_feature_cols()
    trading_dates = np.sort(panel["date"].unique())
    cutoff = known_label_cutoff(trading_dates, as_of, horizon)
    embargo_days = horizon + 1 if embargo_days is None else embargo_days

    train_pool = panel[
        (panel["date"] <= cutoff)
        & panel[TARGET_COLUMN].notna()
    ].dropna(subset=feature_cols + [TARGET_COLUMN]).copy()

    all_dates = np.sort(train_pool["date"].unique())
    min_needed = val_days + embargo_days + 40
    if len(all_dates) < min_needed:
        raise RuntimeError(f"only {len(all_dates)} training dates; need at least {min_needed}")

    val_start = pd.Timestamp(all_dates[-val_days])
    train_end = pd.Timestamp(all_dates[-(val_days + embargo_days + 1)])

    train_df = train_pool[train_pool["date"] <= train_end].copy()
    val_df = train_pool[train_pool["date"] >= val_start].copy()
    return train_df, val_df, cutoff, train_end, val_start


def rank_ic(y_true: np.ndarray, y_pred: np.ndarray, dates: np.ndarray) -> float:
    ics: list[float] = []
    for d in np.unique(dates):
        mask = dates == d
        if mask.sum() < 30:
            continue
        rho, _ = spearmanr(y_true[mask], y_pred[mask])
        if not np.isnan(rho):
            ics.append(float(rho))
    return float(np.mean(ics)) if ics else float("nan")


def train_one(train_df: pd.DataFrame, val_df: pd.DataFrame, seed: int) -> xgb.XGBRegressor:
    feature_cols = get_feature_cols()
    model = xgb.XGBRegressor(
        objective="reg:squarederror",
        n_estimators=700,
        max_depth=3,
        learning_rate=0.025,
        subsample=0.70,
        colsample_bytree=0.80,
        min_child_weight=35,
        reg_alpha=0.10,
        reg_lambda=8.0,
        tree_method="hist",
        n_jobs=-1,
        early_stopping_rounds=50,
        random_state=seed,
    )
    model.fit(
        train_df[feature_cols],
        train_df[TARGET_COLUMN],
        eval_set=[(val_df[feature_cols], val_df[TARGET_COLUMN])],
        verbose=False,
    )
    return model


def train_models(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    n_seeds: int = DEFAULT_N_SEEDS,
) -> tuple[list[xgb.XGBRegressor], float]:
    feature_cols = get_feature_cols()
    y_val = val_df[TARGET_COLUMN].to_numpy()
    d_val = val_df["date"].to_numpy()
    models: list[xgb.XGBRegressor] = []
    preds: list[np.ndarray] = []

    for seed in range(n_seeds):
        model = train_one(train_df, val_df, seed=seed)
        pred = model.predict(val_df[feature_cols])
        models.append(model)
        preds.append(pred)
        print(f"   seed {seed}: val Rank IC = {rank_ic(y_val, pred, d_val):+.4f}")

    avg_pred = np.mean(preds, axis=0)
    ensemble_ic = rank_ic(y_val, avg_pred, d_val)
    print(f"   ensemble val Rank IC = {ensemble_ic:+.4f}")
    return models, ensemble_ic


def _cs_rank(s: pd.Series) -> pd.Series:
    return s.rank(pct=True, method="average") - 0.5


def _rule_score(pred_df: pd.DataFrame, mode: str) -> pd.Series:
    if mode == "liquidity":
        score = pred_df["turnover_ma_20d_rank"]
    elif mode == "ma-revert-liq":
        score = (
            -0.30 * pred_df["close_over_ma20_rank"]
            -0.20 * pred_df["close_over_ma60_rank"]
            +0.50 * pred_df["turnover_ma_20d_rank"]
        )
    elif mode == "overext-liq":
        score = (
            -0.30 * pred_df["ret_5d_rank"]
            -0.35 * pred_df["ret_20d_rank"]
            -0.20 * pred_df["close_over_ma20_rank"]
            +0.15 * pred_df["turnover_ma_20d_rank"]
        )
    elif mode == "reversal-liq":
        score = (
            -0.45 * pred_df["ret_20d_rank"]
            -0.25 * pred_df["ret_5d_rank"]
            +0.30 * pred_df["turnover_ma_20d_rank"]
        )
    elif mode == "blend-liq-rev":
        # Diversify away from pure liquidity (closer in spirit to the class ensemble).
        liq = pred_df["turnover_ma_20d_rank"]
        rev_raw = (
            -0.45 * pred_df["ret_20d_rank"]
            -0.25 * pred_df["ret_5d_rank"]
            +0.30 * pred_df["turnover_ma_20d_rank"]
        )
        score = 0.5 * liq + 0.5 * _cs_rank(rev_raw)
    else:
        raise ValueError(f"unknown rule alpha mode: {mode}")
    return _cs_rank(score)


def model_required(alpha_mode: str, model_weight: float) -> bool:
    return alpha_mode == "model" or (alpha_mode == "hybrid" and model_weight > 0.0)


def add_scores(
    pred_df: pd.DataFrame,
    models: list[xgb.XGBRegressor] | None,
    model_weight: float,
    alpha_mode: str,
) -> pd.DataFrame:
    feature_cols = get_feature_cols()
    pred_df = pred_df.copy()

    if alpha_mode in {
        "liquidity",
        "ma-revert-liq",
        "overext-liq",
        "reversal-liq",
        "blend-liq-rev",
    }:
        pred_df["model_score"] = np.nan
        pred_df["model_score_rank"] = 0.0
        pred_df["prior_score_rank"] = _rule_score(pred_df, alpha_mode)
        pred_df["final_score"] = pred_df["prior_score_rank"]
        return pred_df

    if not models:
        raise ValueError(f"alpha_mode={alpha_mode} requires trained models")

    model_pred = np.mean([m.predict(pred_df[feature_cols]) for m in models], axis=0)
    pred_df["model_score"] = model_pred
    pred_df["model_score_rank"] = _cs_rank(pred_df["model_score"])

    if alpha_mode == "model":
        pred_df["prior_score_rank"] = 0.0
        pred_df["final_score"] = pred_df["model_score_rank"]
    elif alpha_mode == "hybrid":
        pred_df["prior_score_rank"] = _rule_score(pred_df, "overext-liq")
        pred_df["final_score"] = (
            model_weight * pred_df["model_score_rank"]
            + (1.0 - model_weight) * pred_df["prior_score_rank"]
        )
    else:
        raise ValueError(f"unknown alpha_mode: {alpha_mode}")
    return pred_df


def build_portfolio(pred_df: pd.DataFrame, cfg: PortfolioConfig) -> pd.Series:
    if cfg.top_k < MIN_STOCKS:
        raise ValueError(f"top_k must be >= {MIN_STOCKS}")
    if not 0.0 <= cfg.equal_weight_share <= 1.0:
        raise ValueError("equal_weight_share must be in [0, 1]")

    indexed = pred_df.set_index("stock_code").copy()
    eligible = pd.Series(True, index=indexed.index)
    for col in ["suspend_flag", "no_trade_today", "limit_up_yday", "limit_up_today"]:
        if col in indexed.columns:
            eligible &= indexed[col].fillna(0.0) < 1.0

    if "turnover_ma_20d_rank" in indexed.columns:
        eligible &= indexed["turnover_ma_20d_rank"] > (cfg.liquidity_floor_pct - 0.5)

    pool = indexed.loc[eligible].copy()
    if len(pool) < cfg.top_k:
        print(f"   only {len(pool)} eligible names; relaxing filters to preserve constraints")
        pool = indexed.copy()

    chosen = pool.sort_values("final_score", ascending=False).head(cfg.top_k)
    n = len(chosen)
    if n < MIN_STOCKS:
        raise RuntimeError(f"only {n} selected names")

    equal_w = pd.Series(1.0 / n, index=chosen.index)
    ranks = pd.Series(np.arange(n, 0, -1, dtype=float), index=chosen.index)
    rank_w = ranks / ranks.sum()
    w = cfg.equal_weight_share * equal_w + (1.0 - cfg.equal_weight_share) * rank_w

    for _ in range(50):
        over = w > cfg.max_weight
        if not over.any():
            break
        excess = (w[over] - cfg.max_weight).sum()
        w[over] = cfg.max_weight
        free = ~over
        if not free.any():
            break
        w[free] += excess * w[free] / w[free].sum()

    w = w / w.sum()
    assert abs(w.sum() - 1.0) < 1e-8
    assert (w <= cfg.max_weight + 1e-9).all()
    assert (w > 0).sum() >= MIN_STOCKS
    return w


def score_portfolio(
    weights: pd.Series,
    prices: pd.DataFrame,
    index_df: pd.DataFrame,
    as_of: pd.Timestamp,
    horizon: int,
) -> dict[str, float | str | int]:
    prices = prices.copy()
    prices["date"] = pd.to_datetime(prices["date"])
    index_df = index_df.copy()
    index_df["date"] = pd.to_datetime(index_df["date"])
    trading_dates = np.sort(index_df["date"].unique())
    as_of = resolve_asof(trading_dates, as_of)
    asof_idx = int(np.searchsorted(trading_dates, np.datetime64(as_of)))
    if asof_idx + horizon >= len(trading_dates):
        raise ValueError(f"not enough future index data after {as_of.date()} for horizon={horizon}")
    exit_date = pd.Timestamp(trading_dates[asof_idx + horizon])

    close = prices.pivot(index="date", columns="stock_code", values="close")
    entry = close.reindex(index=[as_of]).iloc[0].reindex(weights.index)
    exit_ = close.reindex(index=[exit_date]).iloc[0].reindex(weights.index)
    stock_rets = (exit_ / entry - 1.0).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    portfolio_return = float((weights * stock_rets).sum())

    idx = index_df.set_index("date").sort_index()
    benchmark_return = float(idx.loc[exit_date, "close"] / idx.loc[as_of, "close"] - 1.0)

    return {
        "as_of": as_of.date().isoformat(),
        "exit_date": exit_date.date().isoformat(),
        "horizon": horizon,
        "portfolio_return": portfolio_return,
        "benchmark_return": benchmark_return,
        "excess_return": portfolio_return - benchmark_return,
    }


def generate_weights(
    prices: pd.DataFrame,
    index_df: pd.DataFrame,
    as_of: str | pd.Timestamp,
    horizon: int,
    n_seeds: int,
    val_days: int,
    cfg: PortfolioConfig,
) -> tuple[pd.Series, pd.DataFrame, dict[str, str | int | float]]:
    panel = build_feature_panel(prices, index_df, horizon=horizon)
    trading_dates = np.sort(panel["date"].unique())
    as_of_ts = resolve_asof(trading_dates, as_of)
    cutoff = known_label_cutoff(trading_dates, as_of_ts, horizon)
    models: list[xgb.XGBRegressor] | None = None
    val_ic = float("nan")

    if model_required(cfg.alpha_mode, cfg.model_weight):
        train_df, val_df, cutoff, train_end, val_start = make_train_val(
            panel, as_of=as_of_ts, horizon=horizon, val_days=val_days
        )
        print(f"   known-label cutoff: {cutoff.date()}")
        print(f"   train: {len(train_df):,} rows <= {train_end.date()}")
        print(f"   val:   {len(val_df):,} rows >= {val_start.date()}")
        models, val_ic = train_models(train_df, val_df, n_seeds=n_seeds)
    else:
        print(f"   alpha_mode={cfg.alpha_mode}; no model training needed")
        print(f"   known-label cutoff if training were used: {cutoff.date()}")

    pred_df = panel[panel["date"] == as_of_ts].dropna(subset=get_feature_cols()).copy()
    if pred_df.empty:
        raise RuntimeError(f"empty prediction frame for as_of={as_of_ts.date()}")
    pred_df = add_scores(
        pred_df,
        models,
        model_weight=cfg.model_weight,
        alpha_mode=cfg.alpha_mode,
    )
    weights = build_portfolio(pred_df, cfg)

    meta = {
        "as_of": as_of_ts.date().isoformat(),
        "horizon": horizon,
        "n_names": int((weights > 0).sum()),
        "max_weight": float(weights.max()),
        "val_rank_ic": float(val_ic),
        "known_label_cutoff": cutoff.date().isoformat(),
        "alpha_mode": cfg.alpha_mode,
    }
    return weights, pred_df, meta


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prices", default=str(DATA_DIR / "prices.parquet"))
    parser.add_argument("--index", default=str(DATA_DIR / "index.parquet"))
    parser.add_argument("--as-of", required=True, help="required; e.g. 20260508")
    parser.add_argument("--horizon", type=int, default=5, choices=[3, 5])
    parser.add_argument("--top-k", type=int, default=DEFAULT_TOP_K)
    parser.add_argument("--n-seeds", type=int, default=DEFAULT_N_SEEDS)
    parser.add_argument("--val-days", type=int, default=DEFAULT_VAL_DAYS)
    parser.add_argument("--equal-weight-share", type=float, default=0.70)
    parser.add_argument("--model-weight", type=float, default=0.60)
    parser.add_argument("--liquidity-floor-pct", type=float, default=0.10)
    parser.add_argument(
        "--alpha-mode",
        default=DEFAULT_ALPHA_MODE,
        choices=[
            "liquidity",
            "ma-revert-liq",
            "overext-liq",
            "reversal-liq",
            "blend-liq-rev",
            "model",
            "hybrid",
        ],
        help="pure rule alpha by default; use hybrid/model for XGBoost scoring",
    )
    parser.add_argument("--out", default="submission_v3.csv")
    args = parser.parse_args()

    prices = pd.read_parquet(args.prices)
    index_df = pd.read_parquet(args.index)
    print(
        f">> loaded {len(prices):,} price rows, "
        f"{prices['stock_code'].nunique()} stocks"
    )
    print(f">> building V3 robust model as_of={args.as_of}, horizon={args.horizon}")

    cfg = PortfolioConfig(
        top_k=args.top_k,
        equal_weight_share=args.equal_weight_share,
        liquidity_floor_pct=args.liquidity_floor_pct,
        model_weight=args.model_weight,
        alpha_mode=args.alpha_mode,
    )
    weights, pred_df, meta = generate_weights(
        prices=prices,
        index_df=index_df,
        as_of=args.as_of,
        horizon=args.horizon,
        n_seeds=args.n_seeds,
        val_days=args.val_days,
        cfg=cfg,
    )

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out = pd.DataFrame({"stock_code": weights.index, "weight": weights.values})
    out.to_csv(out_path, index=False)

    indexed = pred_df.set_index("stock_code")
    print(">> top holdings")
    for code, weight in weights.sort_values(ascending=False).head(12).items():
        row = indexed.loc[code]
        print(
            f"   {code} weight={weight:.4f} "
            f"final={row['final_score']:+.4f} model={row['model_score_rank']:+.4f} "
            f"prior={row['prior_score_rank']:+.4f}"
        )

    print(f">> wrote {len(out)} names -> {out_path}")
    print(
        f"   sum={out['weight'].sum():.8f}, max={out['weight'].max():.4f}, "
        f"alpha_mode={meta['alpha_mode']}, val_ic={meta['val_rank_ic']:+.4f}, "
        f"cutoff={meta['known_label_cutoff']}"
    )


if __name__ == "__main__":
    main()
