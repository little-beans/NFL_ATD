import argparse
import sys
from pathlib import Path
import pandas as pd
import nflreadpy as nfl

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from td_engine.current_roster import normalize_weekly_roster


def _pd(obj):
    return obj.to_pandas() if hasattr(obj, "to_pandas") else pd.DataFrame(obj)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--season", type=int, required=True)
    p.add_argument("--week", type=int, required=True)
    args = p.parse_args()

    print(f"Loading weekly rosters for {args.season} Week {args.week}...")
    raw = _pd(nfl.load_rosters_weekly([args.season]))
    print(f"Loaded {len(raw):,} raw weekly-roster rows")

    roster = normalize_weekly_roster(raw, args.season, args.week)
    print(f"Normalized current roster: {len(roster):,} players")

    out = ROOT / "data" / "processed" / f"roster_{args.season}_week_{args.week}.parquet"
    out.parent.mkdir(parents=True, exist_ok=True)
    roster.to_parquet(out, index=False)
    print(f"Saved to {out}")


if __name__ == "__main__":
    main()
