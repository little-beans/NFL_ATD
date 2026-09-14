import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from td_engine.projected_roles import latest_pregame_role_state


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--season", type=int, required=True)
    p.add_argument("--week", type=int, required=True)
    args = p.parse_args()

    path = ROOT/"data"/"processed"/"td_pregame_features_xtd.parquet"
    if not path.exists():
        path = ROOT/"data"/"processed"/"td_pregame_features.parquet"

    d = pd.read_parquet(path)
    latest = latest_pregame_role_state(d,args.season,args.week)

    print(f"Source: {path}")
    print(f"Latest player role states: {len(latest):,}")

    for c in [
        "red_zone_share",
        "volume_share",
        "goal_line_share",
        "target_share",
        "td_efficiency_score",
    ]:
        if c in latest.columns:
            s = pd.to_numeric(latest[c],errors="coerce")
            print(
                f"{c}: non-null={s.notna().sum():,}, "
                f"mean={s.mean():.4f}, median={s.median():.4f}, "
                f"max={s.max():.4f}"
            )


if __name__ == "__main__":
    main()
