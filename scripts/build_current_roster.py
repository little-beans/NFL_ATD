import argparse
import sys
from pathlib import Path
import pandas as pd
import nflreadpy as nfl

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from td_engine.current_roster import build_current_roster


def _pd(obj):
    return obj.to_pandas() if hasattr(obj, "to_pandas") else pd.DataFrame(obj)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--season", type=int, required=True)
    p.add_argument("--week", type=int, required=True)
    args = p.parse_args()

    print(f"Loading weekly rosters for {args.season}...")
    weekly = _pd(nfl.load_rosters_weekly([args.season]))
    print(f"Weekly roster rows: {len(weekly):,}")

    print(f"Loading season roster for {args.season}...")
    season_roster = _pd(nfl.load_rosters([args.season]))
    print(f"Season roster rows: {len(season_roster):,}")

    roster = build_current_roster(
        weekly_rosters=weekly,
        season_rosters=season_roster,
        season=args.season,
        week=args.week,
    )

    print(f"Resolved current roster: {len(roster):,} players")

    if not roster.empty and "roster_source" in roster.columns:
        print("\nRoster source counts:")
        print(roster["roster_source"].value_counts(dropna=False).to_string())

    out = ROOT / "data" / "processed" / f"roster_{args.season}_week_{args.week}.parquet"
    out.parent.mkdir(parents=True, exist_ok=True)
    roster.to_parquet(out, index=False)

    print(f"\nSaved to {out}")


if __name__ == "__main__":
    main()
