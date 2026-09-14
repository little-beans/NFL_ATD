import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from td_engine.odds_value import (
    match_prices_to_model,
    build_value_board,
    best_price_board,
)
from td_engine.value_quality import (
    add_value_quality,
    sort_value_board,
)
from td_engine.console_view import print_value_console


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--season", type=int, required=True)
    p.add_argument("--week", type=int, required=True)

    p.add_argument("--min-edge", type=float, default=2.0)
    p.add_argument("--min-roi", type=float, default=0.0)
    p.add_argument("--min-books", type=int, default=2)
    p.add_argument("--min-role-support", type=float, default=35.0)
    p.add_argument("--max-dispersion", type=float, default=7.5)

    p.add_argument(
        "--top",
        type=int,
        default=10,
        help="Rows shown per console section.",
    )
    p.add_argument(
        "--verbose",
        action="store_true",
        help="Print extra diagnostic details.",
    )

    args = p.parse_args()

    model_path = (
        ROOT / "data" / "output" /
        f"{args.season}_week_{args.week}_td_unified_rankings.csv"
    )
    odds_path = (
        ROOT / "data" / "processed" /
        f"atd_odds_{args.season}_week_{args.week}.parquet"
    )

    if not model_path.exists():
        raise FileNotFoundError(model_path)
    if not odds_path.exists():
        raise FileNotFoundError(
            f"{odds_path}\nRun fetch_atd_odds.py first."
        )

    model = pd.read_csv(model_path)
    prices = pd.read_parquet(odds_path)

    if (
        "fair_american_odds" in model.columns
        and "model_fair_odds" not in model.columns
    ):
        model["model_fair_odds"] = model["fair_american_odds"]

    matched, unmatched = match_prices_to_model(model, prices)
    values = build_value_board(matched)
    best = best_price_board(values)

    best = add_value_quality(
        best,
        values,
        min_books_actionable=args.min_books,
        min_role_support=args.min_role_support,
        max_dispersion_pp=args.max_dispersion,
        min_edge_pp=args.min_edge,
        min_roi_pct=args.min_roi,
    )
    best = sort_value_board(best)

    out_dir = ROOT / "data" / "output"
    out_dir.mkdir(parents=True, exist_ok=True)

    all_path = out_dir / f"{args.season}_week_{args.week}_atd_all_book_prices.csv"
    best_path = out_dir / f"{args.season}_week_{args.week}_atd_value_board.csv"
    actionable_path = out_dir / f"{args.season}_week_{args.week}_atd_actionable.csv"
    review_path = out_dir / f"{args.season}_week_{args.week}_atd_review_only.csv"
    unmatched_path = out_dir / f"{args.season}_week_{args.week}_atd_unmatched_prices.csv"

    values.to_csv(all_path, index=False)
    best.to_csv(best_path, index=False)
    unmatched.to_csv(unmatched_path, index=False)

    actionable = best[best["value_quality"].eq("ACTIONABLE")].copy()
    review = best[best["value_quality"].eq("REVIEW_ONLY")].copy()

    actionable.to_csv(actionable_path, index=False)
    review.to_csv(review_path, index=False)

    if args.verbose:
        print("Value-quality thresholds:")
        print(f"  min books:         {args.min_books}")
        print(f"  min edge:          {args.min_edge:.2f} pp")
        print(f"  min expected ROI:  {args.min_roi:.2f}%")
        print(f"  min role support:  {args.min_role_support:.2f}")
        print(f"  max dispersion:    {args.max_dispersion:.2f} pp")
        print()
        print(f"Sportsbook ATD rows: {len(prices):,}")
        print(f"Matched sportsbook rows: {len(matched):,}")
        print(f"Unmatched sportsbook rows: {len(unmatched):,}")

    print_value_console(
        best,
        matched_count=len(matched),
        unmatched_count=len(unmatched),
        top_n=args.top,
    )

    print()
    print("Files:")
    print(f"  Full value board : {best_path}")
    print(f"  All book prices  : {all_path}")
    print(f"  Unmatched prices : {unmatched_path}")

    if args.verbose:
        print(f"  Quality actionable: {actionable_path}")
        print(f"  Quality review    : {review_path}")


if __name__ == "__main__":
    main()
