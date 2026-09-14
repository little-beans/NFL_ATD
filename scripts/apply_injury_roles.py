import argparse
import sys
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from td_engine.role_reallocation import apply_role_reallocation

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--season", type=int, required=True)
    p.add_argument("--week", type=int, required=True)
    args = p.parse_args()

    p1 = ROOT / "data" / "output" / f"{args.season}_week_{args.week}_td_matchup_rankings.parquet"
    p2 = ROOT / "data" / "output" / f"{args.season}_week_{args.week}_td_matchup_rankings.csv"
    rankings_path = p1 if p1.exists() else p2
    injury_path = ROOT / "data" / "processed" / f"injuries_{args.season}_week_{args.week}.parquet"

    if not rankings_path.exists():
        raise FileNotFoundError(f"Missing weekly matchup rankings: {rankings_path}")
    if not injury_path.exists():
        raise FileNotFoundError(f"Missing injury context: {injury_path}")

    rows = pd.read_parquet(rankings_path) if rankings_path.suffix.lower()==".parquet" else pd.read_csv(rankings_path)
    injuries = pd.read_parquet(injury_path)

    print(f"Loaded {len(rows):,} weekly player rows")
    print(f"Loaded {len(injuries):,} injury rows")

    # Backward compatibility if older scorer omitted player_id.
    if "player_id" not in rows.columns:
        hist_path = ROOT / "data" / "processed" / "td_pregame_features.parquet"
        if hist_path.exists() and "player_name" in rows.columns and "posteam" in rows.columns:
            hist = pd.read_parquet(hist_path)
            latest = (
                hist.sort_values(["season","week"])
                .drop_duplicates(["player_name","posteam"], keep="last")
            )
            map_cols = [c for c in ["player_name","posteam","player_id"] if c in latest.columns]
            latest = latest[map_cols].drop_duplicates(["player_name","posteam"])
            rows = rows.merge(latest, on=["player_name","posteam"], how="left")
            print(f"Reconstructed player_id for {rows['player_id'].notna().sum():,}/{len(rows):,} rows")
        else:
            raise KeyError("Weekly rankings missing player_id and could not reconstruct it.")

    inj_keep = [c for c in ["player_id","report_status","practice_status","availability_multiplier","source_week","source_week_fallback"] if c in injuries.columns]

    rows = rows.drop(columns=[c for c in inj_keep if c!="player_id" and c in rows.columns], errors="ignore")
    rows = rows.merge(injuries[inj_keep], on="player_id", how="left")
    rows["availability_multiplier"] = rows["availability_multiplier"].fillna(1.0)
    if "report_status" not in rows.columns:
        rows["report_status"] = ""
    else:
        rows["report_status"] = rows["report_status"].fillna("")

    adjusted = apply_role_reallocation(rows)

    out = ROOT / "data" / "output" / f"{args.season}_week_{args.week}_td_injury_roles.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    adjusted.to_csv(out, index=False)

    cols = [c for c in [
        "player_name","posteam","position","report_status","source_week",
        "availability_multiplier","injury_role_boost","injury_context_tag",
        "volume_share","adjusted_volume_share",
        "target_share","adjusted_target_share",
        "red_zone_share","adjusted_red_zone_share",
        "goal_line_share","adjusted_goal_line_share",
        "td_probability_pct","td_probability"
    ] if c in adjusted.columns]

    print()
    print(adjusted.sort_values("injury_role_boost", ascending=False)[cols].head(30).to_string(index=False))
    print(f"\nSaved injury-adjusted role table to {out}")

if __name__ == "__main__":
    main()
