import argparse
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--season", type=int, required=True)
    p.add_argument("--week", type=int, required=True)
    args = p.parse_args()

    processed = ROOT / "data" / "processed"

    preferred = [
        processed / "xtd_player_games.parquet",
        processed / "td_pregame_features_xtd.parquet",
        processed / "td_xtd_history.parquet",
    ]

    paths = [p for p in preferred if p.exists()]

    if not paths:
        paths = sorted(processed.glob("*.parquet"))

    if not paths:
        raise FileNotFoundError("No parquet files found in data/processed.")

    found = False

    for path in paths:
        d = pd.read_parquet(path)

        interesting = [
            c for c in d.columns
            if any(k in c.lower() for k in [
                "xtd", "x_td", "expected_td", "actual_td",
                "td_debt", "scored_td", "touchdown"
            ])
        ]

        if not interesting:
            continue

        found = True
        print(f"\nFILE: {path}")
        print(f"Rows: {len(d):,}")
        print("xTD-like columns:")
        for c in interesting:
            s = pd.to_numeric(d[c], errors="coerce")
            if s.notna().any():
                print(
                    f"  {c}: non-null={s.notna().sum():,}, "
                    f"non-zero={s.fillna(0).ne(0).sum():,}, "
                    f"sum={s.fillna(0).sum():.3f}"
                )
            else:
                print(f"  {c}: non-numeric")

        if "season" in d.columns and "week" in d.columns:
            season = pd.to_numeric(d["season"], errors="coerce")
            week = pd.to_numeric(d["week"], errors="coerce")
            eligible = d[
                season.lt(args.season)
                | (season.eq(args.season) & week.lt(args.week))
            ]
            print(
                f"Rows eligible before {args.season} Week {args.week}: "
                f"{len(eligible):,}"
            )

    if not found:
        print("No xTD-like columns found in processed parquet files.")


if __name__ == "__main__":
    main()
