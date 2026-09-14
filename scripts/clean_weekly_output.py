import argparse
import sys
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from td_engine.current_roster import apply_current_roster
from td_engine.probability_guardrails import add_probability_diagnostics


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--season", type=int, required=True)
    p.add_argument("--week", type=int, required=True)
    args = p.parse_args()

    candidates = [
        ROOT/"data"/"output"/f"{args.season}_week_{args.week}_td_injury_roles.csv",
        ROOT/"data"/"output"/f"{args.season}_week_{args.week}_td_matchup_rankings.csv",
    ]
    source = next((p for p in candidates if p.exists()), None)
    if source is None:
        raise FileNotFoundError("No weekly injury/matchup output found.")

    roster_path = ROOT/"data"/"processed"/f"roster_{args.season}_week_{args.week}.parquet"
    if not roster_path.exists():
        raise FileNotFoundError(
            f"Missing {roster_path}. Run build_current_roster.py first."
        )

    rows = pd.read_csv(source)
    roster = pd.read_parquet(roster_path)

    if "player_id" not in rows.columns:
        hist_path = ROOT/"data"/"processed"/"td_pregame_features.parquet"
        hist = pd.read_parquet(hist_path)
        latest = (
            hist.sort_values(["season","week"])
            .drop_duplicates(["player_name","posteam"], keep="last")
            [["player_name","posteam","player_id"]]
        )
        rows = rows.merge(latest, on=["player_name","posteam"], how="left")

    cleaned = apply_current_roster(rows, roster)

    prob_col = "td_probability"
    if prob_col not in cleaned.columns and "td_probability_pct" in cleaned.columns:
        cleaned["td_probability"] = (
            pd.to_numeric(cleaned["td_probability_pct"], errors="coerce") / 100.0
        )

    cleaned = add_probability_diagnostics(cleaned, "td_probability")
    if "td_probability" in cleaned.columns:
        cleaned["td_probability_pct"] = 100 * cleaned["td_probability"]

    out = ROOT/"data"/"output"/f"{args.season}_week_{args.week}_td_cleaned.csv"
    cleaned.to_csv(out, index=False)

    print(f"Rows: {len(cleaned):,}")
    print(f"Roster matched: {cleaned['roster_matched'].sum():,}/{len(cleaned):,}")
    print(f"Team changes detected: {cleaned['team_changed'].sum():,}")
    if "probability_clipped" in cleaned.columns:
        print(f"Extreme probabilities clipped: {cleaned['probability_clipped'].sum():,}")

    changed = cleaned[cleaned["team_changed"]].copy()
    if not changed.empty:
        show = [c for c in [
            "player_name","historical_team","current_team","position"
        ] if c in changed.columns]
        print("\nTeam corrections:")
        print(changed[show].head(30).to_string(index=False))

    unmatched = cleaned[~cleaned["roster_matched"]]
    if not unmatched.empty:
        show = [c for c in ["player_name","player_id","historical_team"] if c in unmatched.columns]
        print("\nUNMATCHED roster players:")
        print(unmatched[show].head(30).to_string(index=False))

    print(f"\nSaved cleaned weekly output to {out}")


if __name__ == "__main__":
    main()
