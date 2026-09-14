import argparse
import sys
from pathlib import Path

import pandas as pd
import nflreadpy as nfl

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from td_engine.injuries import prepare_injury_table

def _to_pandas(obj):
    if hasattr(obj, "to_pandas"):
        return obj.to_pandas()
    return pd.DataFrame(obj)

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--season", type=int, required=True)
    p.add_argument("--week", type=int, required=True)
    args = p.parse_args()

    print(f"Loading nflverse injuries for {args.season} Week {args.week}...")
    injuries = _to_pandas(nfl.load_injuries([args.season]))
    print(f"Loaded {len(injuries):,} injury-report rows")

    table = prepare_injury_table(injuries, args.season, args.week)
    print(f"Target-week/fallback injury rows: {len(table):,}")

    if not table.empty and "source_week_fallback" in table.columns and table["source_week_fallback"].any():
        weeks = sorted(table["source_week"].dropna().unique().tolist())
        print(f"No exact Week {args.week} report rows found; using latest available reports from weeks: {weeks}")

    out = ROOT / "data" / "processed" / f"injuries_{args.season}_week_{args.week}.parquet"
    out.parent.mkdir(parents=True, exist_ok=True)
    table.to_parquet(out, index=False)

    if not table.empty:
        important = table[table["availability_multiplier"] < 1].copy()
        if not important.empty:
            show = [c for c in ["full_name","team","position","report_status","practice_status","source_week","availability_multiplier"] if c in important.columns]
            print("\nPlayers affecting role allocation:")
            print(important[show].to_string(index=False))
        else:
            print("\nNo limited/out players found in selected injury context.")

    print(f"\nSaved injury context to {out}")

if __name__ == "__main__":
    main()
