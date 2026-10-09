"""
walkforward_validate.py -- event-style validation for CSI500 submissions.

This script repeatedly simulates the live competition:
  1. choose a historical as-of date;
  2. train using only labels known by that date;
  3. generate a long-only portfolio at the as-of close;
  4. hold for 3 or 5 trading days;
  5. score excess return versus CSI500.

Use this to compare portfolio rules and model settings. It is intentionally
slower than a single Rank IC split because it measures the thing Gradescope
actually grades: realized excess return from a timestamp-correct portfolio.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from xgboost_v3_robust import (
    DATA_DIR,
    PortfolioConfig,
    add_scores,
    build_feature_panel,
    build_portfolio,
    get_feature_cols,
    known_label_cutoff,
    make_train_val,
    model_required,
    resolve_asof,
    score_portfolio,
    train_models,
)


def _asof_grid(
    trading_dates: np.ndarray,
    start: str | None,
    end: str | None,
    horizon: int,
    step: int,
    min_warmup_days: int,
) -> list[pd.Timestamp]:
    dates = [pd.Timestamp(d) for d in trading_dates]
    first = pd.Timestamp(start) if start else dates[min_warmup_days]
    last = pd.Timestamp(end) if end else dates[-(horizon + 1)]
    first = resolve_asof(trading_dates, first)
    last = resolve_asof(trading_dates, last)

    grid = [
        d for d in dates
        if d >= first and d <= last
        and np.searchsorted(trading_dates, np.datetime64(d)) + horizon < len(trading_dates)
    ]
    return grid[::step]


def _maybe_cap_folds(asofs: list[pd.Timestamp], max_folds: int | None) -> list[pd.Timestamp]:
    if max_folds is None or max_folds <= 0:
        return asofs
    return asofs[: max_folds]


def _summarize(results: pd.DataFrame) -> pd.Series:
    excess = results["excess_return"]
    return pd.Series({
        "folds": len(results),
        "mean_excess_%": excess.mean() * 100,
        "median_excess_%": excess.median() * 100,
        "hit_rate": (excess > 0).mean(),
        "p10_excess_%": excess.quantile(0.10) * 100,
        "p90_excess_%": excess.quantile(0.90) * 100,
        "worst_excess_%": excess.min() * 100,
        "best_excess_%": excess.max() * 100,
        "avg_portfolio_%": results["portfolio_return"].mean() * 100,
        "avg_benchmark_%": results["benchmark_return"].mean() * 100,
        # Rule-only folds leave val_rank_ic as NaN; mean only over trained folds.
        "avg_val_ic": (
            float(np.nanmean(ic))
            if (ic := results["val_rank_ic"].to_numpy(dtype=float)).size
            and np.any(np.isfinite(ic))
            else float("nan")
        ),
    })


def run_walkforward(args: argparse.Namespace) -> pd.DataFrame:
    prices = pd.read_parquet(args.prices)
    prices["date"] = pd.to_datetime(prices["date"])
    index_df = pd.read_parquet(args.index)
    index_df["date"] = pd.to_datetime(index_df["date"])

    print(f">> loading panel once for horizon={args.horizon}")
    panel = build_feature_panel(prices, index_df, horizon=args.horizon)
    trading_dates = np.sort(panel["date"].unique())
    asofs = _asof_grid(
        trading_dates=trading_dates,
        start=args.start,
        end=args.end,
        horizon=args.horizon,
        step=args.step,
        min_warmup_days=args.min_warmup_days,
    )
    if not asofs:
        raise RuntimeError("no as-of dates selected")
    if getattr(args, "max_folds", None):
        asofs = _maybe_cap_folds(asofs, args.max_folds)

    print(
        f">> selected {len(asofs)} folds: {asofs[0].date()} -> {asofs[-1].date()} "
        f"(step={args.step} trading days"
        + (f", max_folds={args.max_folds}" if getattr(args, "max_folds", None) else "")
        + ")"
    )

    cfg = PortfolioConfig(
        top_k=args.top_k,
        equal_weight_share=args.equal_weight_share,
        liquidity_floor_pct=args.liquidity_floor_pct,
        model_weight=args.model_weight,
        alpha_mode=args.alpha_mode,
    )

    rows: list[dict] = []
    feature_cols = get_feature_cols()
    for i, as_of in enumerate(asofs, start=1):
        print(f"\n[{i}/{len(asofs)}] as_of={as_of.date()}")
        cutoff = known_label_cutoff(trading_dates, as_of, args.horizon)
        models = None
        val_ic = float("nan")
        if model_required(args.alpha_mode, args.model_weight):
            train_df, val_df, cutoff, train_end, val_start = make_train_val(
                panel,
                as_of=as_of,
                horizon=args.horizon,
                val_days=args.val_days,
            )
            print(
                f"   cutoff={cutoff.date()} train_end={train_end.date()} "
                f"val_start={val_start.date()}"
            )
            models, val_ic = train_models(train_df, val_df, n_seeds=args.n_seeds)
        else:
            print(f"   cutoff={cutoff.date()} alpha_mode={args.alpha_mode}; skip model training")

        pred_df = panel[panel["date"] == as_of].dropna(subset=feature_cols).copy()
        pred_df = add_scores(
            pred_df,
            models,
            model_weight=args.model_weight,
            alpha_mode=args.alpha_mode,
        )
        weights = build_portfolio(pred_df, cfg)
        score = score_portfolio(weights, prices, index_df, as_of=as_of, horizon=args.horizon)
        score.update({
            "n_names": int((weights > 0).sum()),
            "max_weight": float(weights.max()),
            "val_rank_ic": float(val_ic),
            "known_label_cutoff": cutoff.date().isoformat(),
            "alpha_mode": args.alpha_mode,
        })
        rows.append(score)
        print(
            f"   excess={score['excess_return'] * 100:+.3f}% "
            f"portfolio={score['portfolio_return'] * 100:+.3f}% "
            f"benchmark={score['benchmark_return'] * 100:+.3f}% "
            f"names={score['n_names']}"
        )

    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prices", default=str(DATA_DIR / "prices.parquet"))
    parser.add_argument("--index", default=str(DATA_DIR / "index.parquet"))
    parser.add_argument("--horizon", type=int, default=5, choices=[3, 5])
    parser.add_argument("--start", default=None, help="first as-of date, e.g. 20251101")
    parser.add_argument("--end", default=None, help="last as-of date, e.g. 20260430")
    parser.add_argument("--step", type=int, default=5, help="rebalance every N trading days")
    parser.add_argument(
        "--max-folds",
        type=int,
        default=None,
        help="cap number of windows (first N after step subsample), e.g. 10 for a quick check",
    )
    parser.add_argument("--min-warmup-days", type=int, default=120)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--n-seeds", type=int, default=1)
    parser.add_argument("--val-days", type=int, default=40)
    parser.add_argument("--equal-weight-share", type=float, default=0.70)
    parser.add_argument("--model-weight", type=float, default=0.60)
    parser.add_argument("--liquidity-floor-pct", type=float, default=0.10)
    parser.add_argument(
        "--alpha-mode",
        default="liquidity",
        choices=[
            "liquidity",
            "ma-revert-liq",
            "overext-liq",
            "reversal-liq",
            "blend-liq-rev",
            "model",
            "hybrid",
        ],
    )
    parser.add_argument("--out", default=None, help="optional CSV path for fold-level results")
    args = parser.parse_args()

    results = run_walkforward(args)
    summary = _summarize(results)
    print("\n>> walk-forward summary")
    for key, value in summary.items():
        if isinstance(value, (int, np.integer)):
            print(f"   {key}: {value}")
        else:
            print(f"   {key}: {value:+.4f}")

    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        results.to_csv(out_path, index=False)
        print(f">> wrote fold results -> {out_path}")


if __name__ == "__main__":
    main()
