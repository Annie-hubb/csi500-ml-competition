"""
compare_walkforward_h5.py — 四版本 5 日滚动 walk-forward 超额收益对比

每个 fold：在 as_of 收盘建仓，持有 5 个交易日，用 score_portfolio 算相对 CSI500 的 excess。

四个版本（你要看的四条 excess 曲线）
--------------------------------
  [1] baseline      — 老师给的 baseline_xgboost.py（features.py、单模型、top_k 可调）
  [2] V2            — xgboost_v2.py（60d 特征、train 内 winsor、5 seed、风控+流动性过滤）
  [3] V3 原版       — xgboost_v3_robust.py，alpha_mode=ma-revert-liq（改默认前的「原版 V3 规则」）
  [4] V3 改后       — 同上，alpha_mode=liquidity（你现在默认的改后 V3）

宽表 CSV 列名：excess_baseline, excess_v2, excess_v3_original, excess_v3_tuned
（数值为小数，例如 0.01 = +1% excess）

Note: V3 两条用 V3 的组合构造（等权+rank、涨跌停过滤等）；baseline/V2 用各自脚本里的组合逻辑。

Example
-------
  python compare_walkforward_h5.py --start 20250701 --end 20260508 --step 5 \\
      --out-prefix results/four_versions_h5
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from baseline_xgboost import EMBARGO_DAYS as BASE_EMBARGO_DAYS
from baseline_xgboost import VAL_DAYS as BASE_VAL_DAYS
from baseline_xgboost import build_portfolio as baseline_build_portfolio
from baseline_xgboost import train_model as baseline_train_model
from features import (
    FEATURE_COLUMNS,
    FORWARD_HORIZON,
    TARGET_COLUMN,
    build_features,
    get_feature_cols,
    prediction_frame,
    training_frame,
)
from walkforward_validate import _asof_grid
from xgboost_v2 import (
    EMBARGO_DAYS as V2_EMBARGO_DAYS,
    LIQUIDITY_PCT as V2_LIQUIDITY_PCT,
    N_SEEDS as V2_N_SEEDS,
    VAL_DAYS as V2_VAL_DAYS,
    build_portfolio as v2_build_portfolio,
    train_one as v2_train_one,
    winsorize_per_date,
)
from xgboost_v3_robust import (
    DATA_DIR,
    PortfolioConfig,
    add_scores,
    build_feature_panel,
    build_portfolio as v3_build_portfolio,
    get_feature_cols as v3_get_feature_cols,
    resolve_asof,
    score_portfolio,
)


def _method_summary(excess: pd.Series) -> dict[str, float]:
    x = excess.astype(float)
    return {
        "mean_excess_%": float(x.mean() * 100),
        "median_excess_%": float(x.median() * 100),
        "hit_rate_%": float((x > 0).mean() * 100),
        "worst_excess_%": float(x.min() * 100),
        "best_excess_%": float(x.max() * 100),
    }


def _baseline_fold(
    panel: pd.DataFrame,
    trading_dates: np.ndarray,
    as_of: pd.Timestamp,
    top_k: int,
) -> pd.Series:
    as_of_ts = resolve_asof(trading_dates, as_of)
    as_of_idx = int(np.searchsorted(trading_dates, np.datetime64(as_of_ts)))
    cutoff_idx = max(0, as_of_idx - FORWARD_HORIZON)
    train_cutoff = pd.Timestamp(trading_dates[cutoff_idx])
    train_pool = training_frame(panel, max_date=train_cutoff)

    all_dates = np.sort(train_pool["date"].unique())
    min_needed = BASE_VAL_DAYS + BASE_EMBARGO_DAYS + 20
    if len(all_dates) < min_needed:
        raise RuntimeError(f"baseline: need >= {min_needed} train dates, got {len(all_dates)}")
    val_start = pd.Timestamp(all_dates[-BASE_VAL_DAYS])
    train_end = pd.Timestamp(all_dates[-(BASE_VAL_DAYS + BASE_EMBARGO_DAYS + 1)])
    train_df = train_pool[train_pool["date"] <= train_end]
    val_df = train_pool[train_pool["date"] >= val_start]

    feature_cols = [c for c in FEATURE_COLUMNS if c in panel.columns]
    model = baseline_train_model(train_df, val_df, feature_cols)
    pred_df = prediction_frame(panel, as_of=as_of_ts)
    pred_df = pred_df.assign(score=model.predict(pred_df[feature_cols]))
    scores = pred_df.set_index("stock_code")["score"]
    return baseline_build_portfolio(scores, top_k=top_k)


def _v2_fold(
    panel: pd.DataFrame,
    trading_dates: np.ndarray,
    as_of: pd.Timestamp,
    top_k: int,
    n_seeds: int,
) -> pd.Series:
    as_of_ts = resolve_asof(trading_dates, as_of)
    as_of_idx = int(np.searchsorted(trading_dates, np.datetime64(as_of_ts)))
    cutoff_idx = max(0, as_of_idx - FORWARD_HORIZON)
    train_cutoff = pd.Timestamp(trading_dates[cutoff_idx])
    train_pool = training_frame(panel, max_date=train_cutoff, use_60d=True).copy()
    winsorize_per_date(train_pool, TARGET_COLUMN, pct=0.01)

    all_dates = np.sort(train_pool["date"].unique())
    if len(all_dates) < V2_VAL_DAYS + V2_EMBARGO_DAYS + 20:
        raise RuntimeError("v2: insufficient training dates")
    val_start = pd.Timestamp(all_dates[-V2_VAL_DAYS])
    train_end = pd.Timestamp(all_dates[-(V2_VAL_DAYS + V2_EMBARGO_DAYS + 1)])
    train_df = train_pool[train_pool["date"] <= train_end]
    val_df = train_pool[train_pool["date"] >= val_start]

    feature_cols = [c for c in get_feature_cols(use_60d=True) if c in panel.columns]
    models = [v2_train_one(train_df, val_df, feature_cols, seed=s) for s in range(n_seeds)]

    pred_df = prediction_frame(panel, as_of=as_of_ts, use_60d=True)
    pred_X = pred_df[feature_cols]
    preds = np.mean([m.predict(pred_X) for m in models], axis=0)
    pred_df = pred_df.assign(score=preds)
    indexed = pred_df.set_index("stock_code")

    eligible = pd.Series(True, index=indexed.index)
    if "suspend_flag" in indexed.columns:
        eligible &= indexed["suspend_flag"] < 1
    if "volume" in indexed.columns:
        eligible &= indexed["volume"] != 0
    if "limit_up_flag" in indexed.columns:
        eligible &= indexed["limit_up_flag"] < 1
    if "turnover_ma_20d" in indexed.columns:
        thr = indexed["turnover_ma_20d"].quantile(V2_LIQUIDITY_PCT)
        eligible &= indexed["turnover_ma_20d"] >= thr

    scores = indexed["score"]
    return v2_build_portfolio(scores, eligible, top_k=top_k)


def _v3_rule_fold(
    panel_v3: pd.DataFrame,
    trading_dates: np.ndarray,
    as_of: pd.Timestamp,
    alpha_mode: str,
    cfg: PortfolioConfig,
) -> pd.Series:
    as_of_ts = resolve_asof(trading_dates, as_of)
    feature_cols = v3_get_feature_cols()
    pred_df = panel_v3[panel_v3["date"] == as_of_ts].dropna(subset=feature_cols).copy()
    pred_df = add_scores(
        pred_df,
        None,
        model_weight=cfg.model_weight,
        alpha_mode=alpha_mode,
    )
    return v3_build_portfolio(pred_df, cfg)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--prices", default=str(DATA_DIR / "prices.parquet"))
    p.add_argument("--index", default=str(DATA_DIR / "index.parquet"))
    p.add_argument("--start", default="20250701")
    p.add_argument("--end", default="20260508")
    p.add_argument("--step", type=int, default=5)
    p.add_argument("--min-warmup-days", type=int, default=120)
    p.add_argument("--baseline-top-k", type=int, default=50)
    p.add_argument("--v2-top-k", type=int, default=40)
    p.add_argument("--v3-top-k", type=int, default=50)
    p.add_argument("--v3-equal-weight-share", type=float, default=0.70)
    p.add_argument("--v3-liquidity-floor-pct", type=float, default=0.10)
    p.add_argument("--v2-n-seeds", type=int, default=V2_N_SEEDS)
    p.add_argument(
        "--out-prefix",
        default="results/four_versions_h5",
        help="writes {prefix}_wide.csv & {prefix}_summary.csv (set '' to skip files)",
    )
    p.add_argument("--quiet", action="store_true", help="only print legend + final summary table")
    args = p.parse_args()

    prices = pd.read_parquet(args.prices)
    prices["date"] = pd.to_datetime(prices["date"])
    index_df = pd.read_parquet(args.index)
    index_df["date"] = pd.to_datetime(index_df["date"])

    horizon = 5
    print(">> building feature panels (one-time)")
    panel_base = build_features(prices, use_60d=False)
    panel_v2 = build_features(prices, use_60d=True)
    panel_v3 = build_feature_panel(prices, index_df, horizon=horizon)

    trading_dates = np.sort(panel_v3["date"].unique())
    asofs = _asof_grid(
        trading_dates=trading_dates,
        start=args.start,
        end=args.end,
        horizon=horizon,
        step=args.step,
        min_warmup_days=args.min_warmup_days,
    )
    if not asofs:
        raise SystemExit("no as-of dates in grid")

    cfg_mr = PortfolioConfig(
        top_k=args.v3_top_k,
        equal_weight_share=args.v3_equal_weight_share,
        liquidity_floor_pct=args.v3_liquidity_floor_pct,
        alpha_mode="ma-revert-liq",
    )
    cfg_liq = PortfolioConfig(
        top_k=args.v3_top_k,
        equal_weight_share=args.v3_equal_weight_share,
        liquidity_floor_pct=args.v3_liquidity_floor_pct,
        alpha_mode="liquidity",
    )

    rows: list[dict] = []
    if not args.quiet:
        print(
            ">> 四版本: [1] baseline (baseline_xgboost)  [2] V2  [3] V3原版(ma-revert-liq)  [4] V3改后(liquidity)"
        )
    print(f">> {len(asofs)} folds: {asofs[0].date()} -> {asofs[-1].date()} (h={horizon}, step={args.step})")

    for i, as_of in enumerate(asofs, start=1):
        if not args.quiet:
            print(f"\n[{i}/{len(asofs)}] as_of={as_of.date()}")
        out: dict = {"as_of": as_of.date().isoformat()}

        w_b = _baseline_fold(panel_base, trading_dates, as_of, top_k=args.baseline_top_k)
        s_b = score_portfolio(w_b, prices, index_df, as_of=as_of, horizon=horizon)
        out["excess_baseline"] = s_b["excess_return"]
        if not args.quiet:
            print(f"   [1] baseline   excess={s_b['excess_return'] * 100:+.3f}%")

        w_v2 = _v2_fold(panel_v2, trading_dates, as_of, top_k=args.v2_top_k, n_seeds=args.v2_n_seeds)
        s_v2 = score_portfolio(w_v2, prices, index_df, as_of=as_of, horizon=horizon)
        out["excess_v2"] = s_v2["excess_return"]
        if not args.quiet:
            print(f"   [2] V2         excess={s_v2['excess_return'] * 100:+.3f}%")

        w_mr = _v3_rule_fold(panel_v3, trading_dates, as_of, "ma-revert-liq", cfg_mr)
        s_mr = score_portfolio(w_mr, prices, index_df, as_of=as_of, horizon=horizon)
        out["excess_v3_original"] = s_mr["excess_return"]
        if not args.quiet:
            print(f"   [3] V3原版     excess={s_mr['excess_return'] * 100:+.3f}%")

        w_lq = _v3_rule_fold(panel_v3, trading_dates, as_of, "liquidity", cfg_liq)
        s_lq = score_portfolio(w_lq, prices, index_df, as_of=as_of, horizon=horizon)
        out["excess_v3_tuned"] = s_lq["excess_return"]
        if not args.quiet:
            print(f"   [4] V3改后     excess={s_lq['excess_return'] * 100:+.3f}%")

        rows.append(out)

    wide = pd.DataFrame(rows)
    summary_rows = []
    for col, label in [
        ("excess_baseline", "1_baseline"),
        ("excess_v2", "2_V2"),
        ("excess_v3_original", "3_V3_original_ma_revert_liq"),
        ("excess_v3_tuned", "4_V3_tuned_liquidity"),
    ]:
        stats = _method_summary(wide[col])
        stats["method"] = label
        summary_rows.append(stats)

    summary = pd.DataFrame(summary_rows).set_index("method")

    print("\n" + "=" * 72)
    print(">> SUMMARY — 每折持有 5 个交易日，超额相对 CSI500（表中 % 为百分数）")
    print("=" * 72)
    print(summary.to_string(float_format=lambda x: f"{x:+.4f}" if abs(x) < 10 else f"{x:+.2f}"))

    if args.out_prefix:
        prefix = Path(args.out_prefix)
        if str(prefix).strip():
            prefix.parent.mkdir(parents=True, exist_ok=True)
            wide_path = prefix.parent / f"{prefix.name}_wide.csv"
            sum_path = prefix.parent / f"{prefix.name}_summary.csv"
            wide.to_csv(wide_path, index=False)
            summary.to_csv(sum_path)
            print(f"\n>> wrote {wide_path}")
            print(f">> wrote {sum_path}")


if __name__ == "__main__":
    main()
