import argparse
import sys
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from td_engine.unified import finalize_weekly_table


def _read_any(path: Path):
    if path.suffix.lower() == ".parquet":
        return pd.read_parquet(path)
    return pd.read_csv(path)


def _find_week_source(season: int, week: int) -> Path:
    candidates = [
        ROOT/"data"/"output"/f"{season}_week_{week}_td_matchup_rankings.parquet",
        ROOT/"data"/"output"/f"{season}_week_{week}_td_matchup_rankings.csv",
        ROOT/"data"/"output"/f"{season}_week_{week}_td_injury_roles.csv",
        ROOT/"data"/"output"/f"{season}_week_{week}_td_cleaned.csv",
    ]
    for p in candidates:
        if p.exists():
            return p
    raise FileNotFoundError(
        "Could not find a weekly matchup ranking file. "
        "Run score_week_matchup.py first."
    )


def _ensure_player_ids(rows: pd.DataFrame) -> pd.DataFrame:
    if "player_id" in rows.columns and rows["player_id"].notna().all():
        return rows

    hist_path = ROOT/"data"/"processed"/"td_pregame_features.parquet"
    if not hist_path.exists():
        raise FileNotFoundError(
            "player_id missing and td_pregame_features.parquet not found."
        )

    hist = pd.read_parquet(hist_path)
    latest = (
        hist.sort_values(["season","week"])
        .drop_duplicates(["player_name","posteam"], keep="last")
        [["player_name","posteam","player_id"]]
        .drop_duplicates(["player_name","posteam"])
    )

    out = rows.merge(
        latest,
        on=["player_name","posteam"],
        how="left",
        suffixes=("","_hist")
    )

    if "player_id_hist" in out.columns:
        if "player_id" not in out.columns:
            out["player_id"] = out["player_id_hist"]
        else:
            out["player_id"] = out["player_id"].fillna(out["player_id_hist"])
        out = out.drop(columns=["player_id_hist"])

    return out


def _latest_xtd_for_players():
    candidates = [
        ROOT/"data"/"processed"/"td_pregame_features_xtd.parquet",
        ROOT/"data"/"processed"/"td_xtd_history.parquet",
    ]
    p = next((p for p in candidates if p.exists()), None)
    if p is None:
        return pd.DataFrame()

    d = pd.read_parquet(p)
    if "player_id" not in d.columns:
        return pd.DataFrame()

    sort_cols = [c for c in ["season","week"] if c in d.columns]
    if sort_cols:
        d = d.sort_values(sort_cols)

    return d.drop_duplicates("player_id", keep="last")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--season", type=int, required=True)
    p.add_argument("--week", type=int, required=True)
    args = p.parse_args()

    source = _find_week_source(args.season, args.week)
    print(f"Loading base weekly matchup rows: {source}")
    rows = _read_any(source)
    rows = _ensure_player_ids(rows)

    roster_path = ROOT/"data"/"processed"/f"roster_{args.season}_week_{args.week}.parquet"
    injury_path = ROOT/"data"/"processed"/f"injuries_{args.season}_week_{args.week}.parquet"

    if not roster_path.exists():
        raise FileNotFoundError(
            f"Missing current roster: {roster_path}\n"
            "Run build_current_roster.py first."
        )

    roster = pd.read_parquet(roster_path)
    injuries = pd.read_parquet(injury_path) if injury_path.exists() else pd.DataFrame()
    xtd = _latest_xtd_for_players()

    print(f"Rows: {len(rows):,}")
    print(f"Roster rows: {len(roster):,}")
    print(f"Injury rows: {len(injuries):,}")
    print(f"xTD player rows: {len(xtd):,}")

    final = finalize_weekly_table(
        rows=rows,
        roster=roster,
        injuries=injuries,
        xtd=xtd,
    )

    # Drop unavailable players from ranked view but preserve in full output.
    rankable = final[
        pd.to_numeric(final["availability_multiplier"], errors="coerce")
        .fillna(1.0)
        .gt(0)
    ].copy()

    if "td_probability" in rankable.columns:
        rankable = rankable.sort_values(
            ["td_probability","injury_adjusted_role_score"],
            ascending=[False,False]
        )

    out_dir = ROOT/"data"/"output"
    out_dir.mkdir(parents=True, exist_ok=True)

    full_path = out_dir/f"{args.season}_week_{args.week}_td_unified_full.csv"
    rank_path = out_dir/f"{args.season}_week_{args.week}_td_unified_rankings.csv"

    final.to_csv(full_path, index=False)
    rankable.to_csv(rank_path, index=False)

    print()
    print(f"Roster matched: {final['roster_matched'].sum():,}/{len(final):,}")
    print(f"Out/Unavailable: {(final['availability_multiplier'] <= 0).sum():,}")
    print(f"Role boosts: {(final['injury_role_boost'] >= 0.03).sum():,}")
    if "probability_clipped" in final.columns:
        print(f"Probability guardrails applied: {final['probability_clipped'].sum():,}")

    show_cols = [
        c for c in [
            "player_name","posteam","opponent","position",
            "td_probability_pct","fair_american_odds",
            "baseline_score","injury_adjusted_role_score",
            "team_implied_points",
            "expected_tds_prior","actual_tds_prior","td_debt",
            "report_status","injury_role_boost","tags"
        ]
        if c in rankable.columns
    ]

    print("\nTop 30 unified rankings:")
    print(rankable[show_cols].head(30).to_string(index=False))

    print(f"\nSaved full table: {full_path}")
    print(f"Saved rankings:  {rank_path}")


if __name__ == "__main__":
    main()
