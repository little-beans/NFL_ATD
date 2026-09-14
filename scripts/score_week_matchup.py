from pathlib import Path
import sys, argparse, joblib

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd
import numpy as np
from score_week import _build_upcoming_rows
from td_engine.matchup import latest_defense_priors
from td_engine.schedule import load_week_team_schedule
from td_engine.x_td import add_td_debt_tags


def _attach_latest_xtd_priors(rows: pd.DataFrame, history: pd.DataFrame) -> pd.DataFrame:
    """Carry the latest completed xTD debt state into an upcoming week."""
    xtd_cols = [
        "expected_tds_prior", "actual_tds_prior", "xtd_opps_prior",
        "expected_tds_l5", "actual_tds_l5", "td_debt_l5", "td_debt",
    ]
    available = [c for c in xtd_cols if c in history.columns]
    if not available:
        rows = rows.copy()
        for c in xtd_cols:
            rows[c] = 0.0
        rows["td_debt_tag"] = ""
        return rows

    sort_cols = [c for c in ["season", "week", "game_id"] if c in history.columns]
    latest = (
        history.sort_values(sort_cols)
        .groupby("player_id", as_index=False)
        .tail(1)[["player_id"] + available]
    )
    out = rows.drop(columns=[c for c in available if c in rows.columns], errors="ignore")
    out = out.merge(latest, on="player_id", how="left", validate="one_to_one")
    for c in xtd_cols:
        if c not in out.columns:
            out[c] = 0.0
        out[c] = out[c].fillna(0.0)
    return add_td_debt_tags(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--season", type=int, required=True)
    ap.add_argument("--week", type=int, required=True)
    ap.add_argument("--data", default="data/processed/td_pregame_features_xtd.parquet")
    ap.add_argument("--model", default="data/processed/td_matchup_model.joblib")
    ap.add_argument("--output", default=None)
    ap.add_argument("--top", type=int, default=40)
    args = ap.parse_args()

    data_path = Path(args.data)
    if not data_path.exists():
        fallback = Path("data/processed/td_pregame_features.parquet")
        print(f"WARNING: {data_path} not found; falling back to {fallback} without xTD debt history.")
        data_path = fallback

    df = pd.read_parquet(data_path)
    model = joblib.load(args.model)
    print(f"Loaded {len(df):,} historical player-games")
    print("Building role priors...", flush=True)
    rows = _build_upcoming_rows(df, args.season, args.week)
    rows = _attach_latest_xtd_priors(rows, df)

    print("Loading target-week schedule and market lines...", flush=True)
    sched = load_week_team_schedule(args.season, args.week)
    if sched.empty:
        raise SystemExit(f"No schedule rows found for {args.season} Week {args.week}")
    print(f"Found {len(sched)//2} games")

    sched_cols = [
        "posteam", "defteam", "game_id", "is_home", "total_line", "team_spread",
        "implied_team_points", "rest_days", "gameday", "gametime",
    ]
    if "posteam" not in rows.columns:
        raise SystemExit("Upcoming player rows are missing 'posteam'.")
    replace_cols = [c for c in sched_cols if c != "posteam" and c in rows.columns]
    rows = rows.drop(columns=replace_cols, errors="ignore")
    schedule_view = sched[[c for c in sched_cols if c in sched.columns]].drop_duplicates("posteam")
    rows = rows.merge(schedule_view, on="posteam", how="inner", validate="many_to_one")

    print("Building opponent defensive priors...", flush=True)
    defense = latest_defense_priors(df, args.season, args.week)
    rows = rows.merge(defense, on="defteam", how="left")

    for c in model.features:
        if c not in rows.columns:
            raise SystemExit(f"Missing model feature: {c}")
        if rows[c].isna().any():
            if c == "total_line":
                rows[c] = rows[c].fillna(44.0)
            elif c == "team_spread":
                rows[c] = rows[c].fillna(0.0)
            elif c == "implied_team_points":
                rows[c] = rows[c].fillna(rows["total_line"] / 2.0)
            else:
                median = rows[c].median()
                rows[c] = rows[c].fillna(0.0 if pd.isna(median) else median)

    scored = model.predict(rows)
    scored["td_probability_pct"] = 100 * scored["td_probability"]
    scored = scored.sort_values(["td_probability", "baseline_score"], ascending=False).reset_index(drop=True)
    scored.insert(0, "rank", np.arange(1, len(scored) + 1))

    keep = [
        "rank", "player_name", "position", "posteam", "defteam", "is_home",
        "baseline_score", "td_probability_pct", "fair_american_odds",
        "implied_team_points", "total_line", "team_spread",
        "def_td_allowed_pg", "def_rz_td_rate",
        "goal_line_share", "red_zone_share", "touches_per_game", "targets_per_game",
        "expected_tds_prior", "actual_tds_prior", "td_debt",
        "expected_tds_l5", "actual_tds_l5", "td_debt_l5", "td_debt_tag",
    ]
    out = scored[[c for c in keep if c in scored.columns]].copy()

    path = Path(args.output) if args.output else Path(
        f"data/output/{args.season}_week_{args.week}_td_matchup_rankings.csv"
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(path, index=False)

    show = out.head(args.top).copy()
    round_cols = [
        "baseline_score", "td_probability_pct", "implied_team_points",
        "def_td_allowed_pg", "def_rz_td_rate", "goal_line_share",
        "expected_tds_prior", "actual_tds_prior", "td_debt",
        "expected_tds_l5", "actual_tds_l5", "td_debt_l5",
    ]
    for c in round_cols:
        if c in show.columns:
            show[c] = show[c].round(2)

    print("\nMATCHUP-AWARE ANYTIME TD RANKINGS + xTD DEBT\n")
    print(show.to_string(index=False))
    print(f"\nSaved {len(out):,} rankings to {path}")
    if "td_debt_tag" in out.columns:
        tagged = out[out["td_debt_tag"] != ""]
        print(f"Tagged regression candidates: {len(tagged)}")


if __name__ == "__main__":
    main()
