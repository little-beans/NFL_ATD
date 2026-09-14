from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Make `python scripts/score_week.py ...` work from the project root on Windows.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import joblib
import numpy as np
import pandas as pd

from td_engine.rolling import build_rolling_pregame_features
from td_engine.score import baseline_score


RAW_ROLLING_COLS = [
    "rush_att", "targets", "receptions", "touches", "td",
    "rz_opp", "inside10_opp", "inside5_opp",
    "rz_carry", "inside10_carry", "inside5_carry",
    "rz_target", "inside10_target", "inside5_target",
    "red_zone_share_game", "inside10_share_game", "goal_line_share_game",
    "volume_share_game", "carry_share_game", "target_share_game",
    "rz_carry_share_game", "inside5_carry_share_game", "rz_target_share_game",
]

SKILL_POSITIONS = {"RB", "WR", "TE", "FB", "HB"}


def parse_args():
    p = argparse.ArgumentParser(
        description="Score an NFL week with the trained Anytime TD role model."
    )
    p.add_argument("--season", type=int, required=True)
    p.add_argument("--week", type=int, required=True)
    p.add_argument("--data", default="data/processed/td_pregame_features.parquet")
    p.add_argument("--model", default="data/processed/td_model.joblib")
    p.add_argument("--output", default=None)
    p.add_argument(
        "--top", type=int, default=50,
        help="Number of ranked players to print to the console."
    )
    return p.parse_args()


def _find_existing_week(df: pd.DataFrame, season: int, week: int) -> pd.DataFrame:
    if "season" not in df.columns or "week" not in df.columns:
        return pd.DataFrame()
    mask = df["season"].eq(season) & df["week"].eq(week)
    if "season_type" in df.columns:
        mask &= df["season_type"].fillna("REG").eq("REG")
    return df.loc[mask].copy()


def _base_history_columns(df: pd.DataFrame) -> list[str]:
    metadata = [
        "season", "week", "season_type", "game_date", "game_id",
        "posteam", "defteam", "player_id", "player_name", "position",
        "scored_td",
    ]
    wanted = metadata + RAW_ROLLING_COLS
    return [c for c in wanted if c in df.columns]


def _latest_player_pool(history: pd.DataFrame, season: int, week: int) -> pd.DataFrame:
    """Players with an appearance before the target week, using their latest known metadata."""
    eligible = history[
        (history["season"] < season)
        | ((history["season"] == season) & (history["week"] < week))
    ].copy()

    if eligible.empty:
        raise SystemExit(f"No history exists before {season} Week {week}.")

    if "season_type" in eligible.columns:
        eligible = eligible[eligible["season_type"].fillna("REG").eq("REG")].copy()

    # For in-season scoring, keep players observed in the target season when possible.
    this_season = eligible[eligible["season"].eq(season)].copy()
    if not this_season.empty:
        eligible = this_season

    sort_cols = [c for c in ["player_id", "season", "week", "game_date", "game_id"] if c in eligible.columns]
    eligible = eligible.sort_values(sort_cols)
    latest = eligible.groupby("player_id", as_index=False, sort=False).tail(1).copy()

    if "position" in latest.columns:
        latest = latest[latest["position"].isin(SKILL_POSITIONS)].copy()

    return latest


def _build_upcoming_rows(df: pd.DataFrame, season: int, week: int) -> pd.DataFrame:
    """
    Build upcoming-week features directly from completed history.

    This avoids rerunning the full rolling engine across every historical row.
    For each player we use:
      1) last-5 completed games,
      2) current-season prior mean as fallback,
      3) career prior mean as fallback.
    TD efficiency uses all prior career touches/TDs with the same smoothing
    as the historical rolling engine.
    """
    history = df[
        (df["season"] < season)
        | ((df["season"] == season) & (df["week"] < week))
    ].copy()

    if "season_type" in history.columns:
        history = history[history["season_type"].fillna("REG").eq("REG")].copy()

    if history.empty:
        raise SystemExit(f"No history exists before {season} Week {week}.")

    latest = _latest_player_pool(history, season, week)
    player_ids = latest["player_id"].dropna().unique().tolist()
    history = history[history["player_id"].isin(player_ids)].copy()

    sort_cols = [c for c in ["player_id", "season", "week", "game_date", "game_id"] if c in history.columns]
    history = history.sort_values(sort_cols)

    print(f"Active player pool: {len(player_ids):,}")
    print(f"Relevant completed history rows: {len(history):,}")
    print("Computing last-5 / season / career priors...", flush=True)

    role_map = {
        "red_zone_share": "red_zone_share_game",
        "volume_share": "volume_share_game",
        "goal_line_share": "goal_line_share_game",
        "target_share": "target_share_game",
        "inside10_share": "inside10_share_game",
        "carry_share": "carry_share_game",
        "rz_carry_share": "rz_carry_share_game",
        "inside5_carry_share": "inside5_carry_share_game",
        "rz_target_share": "rz_target_share_game",
        "touches_per_game": "touches",
        "rush_att_per_game": "rush_att",
        "targets_per_game": "targets",
    }

    # Latest five games per player.
    last5 = history.groupby("player_id", group_keys=False, sort=False).tail(5)
    season_hist = history[history["season"].eq(season)].copy()

    def grouped_mean(frame, col, name):
        if col not in frame.columns or frame.empty:
            return pd.Series(dtype=float, name=name)
        return frame.groupby("player_id")[col].mean().rename(name)

    out = latest.copy().set_index("player_id", drop=False)

    feature_frames = []
    for target_col, source_col in role_map.items():
        l5 = grouped_mean(last5, source_col, f"{target_col}__l5")
        seas = grouped_mean(season_hist, source_col, f"{target_col}__season")
        career = grouped_mean(history, source_col, f"{target_col}__career")
        trio = pd.concat([l5, seas, career], axis=1)
        value = trio.iloc[:, 0] if len(trio.columns) else pd.Series(dtype=float)
        if len(trio.columns) > 1:
            for c in trio.columns[1:]:
                value = value.fillna(trio[c])
        feature_frames.append(value.rename(target_col))

    features = pd.concat(feature_frames, axis=1)

    # `latest` may already contain historical versions of these columns.
    # For an upcoming week we want the newly-computed priors to replace
    # those stale values, not collide with them during join().
    overlap = [c for c in features.columns if c in out.columns]
    if overlap:
        out = out.drop(columns=overlap)

    out = out.join(features, how="left")

    # Career totals before target week for smoothed TD efficiency.
    totals = history.groupby("player_id").agg(
        prior_tds=("td", "sum"),
        prior_touches=("touches", "sum"),
        career_games_before=("game_id", "count"),
    )
    season_counts = season_hist.groupby("player_id").size().rename("season_games_before")

    # Replace any stale copied prior columns before joining freshly computed
    # upcoming-week values. This keeps joins deterministic and avoids
    # pandas overlap errors on fields carried forward from the latest row.
    prior_cols = [
        "prior_tds",
        "prior_touches",
        "career_games_before",
        "season_games_before",
        "pregame_td_rate_smoothed",
        "td_efficiency_score",
    ]
    stale_prior_cols = [c for c in prior_cols if c in out.columns]
    if stale_prior_cols:
        out = out.drop(columns=stale_prior_cols)

    out = out.join(totals, how="left").join(season_counts, how="left")
    out["season_games_before"] = out["season_games_before"].fillna(0).astype(int)

    prior_rate = 0.045
    prior_n = 20.0
    out["pregame_td_rate_smoothed"] = (
        out["prior_tds"].fillna(0) + prior_rate * prior_n
    ) / (out["prior_touches"].fillna(0) + prior_n)
    out["td_efficiency_score"] = 1.0 / (
        1.0 + np.exp(-35.0 * (out["pregame_td_rate_smoothed"] - 0.045))
    )

    out["season"] = season
    out["week"] = week
    if "season_type" in out.columns:
        out["season_type"] = "REG"
    if "game_id" in out.columns:
        out["game_id"] = out["player_id"].astype(str).map(lambda x: f"PROJ_{season}_{week}_{x}")
    if "defteam" in out.columns:
        out["defteam"] = None
    if "scored_td" in out.columns:
        out["scored_td"] = np.nan

    out = out.reset_index(drop=True)
    out = baseline_score(out)
    print("Upcoming-week priors complete.", flush=True)
    return out

def _score(model, rows: pd.DataFrame) -> pd.DataFrame:
    missing = [c for c in model.features if c not in rows.columns]
    if missing:
        raise SystemExit(f"Missing model feature columns: {missing}")

    scored = model.predict(rows)

    keep = [
        "season", "week", "player_id", "player_name", "position",
        "posteam", "defteam", "career_games_before", "season_games_before",
        "baseline_score", "red_zone_share", "volume_share",
        "goal_line_share", "target_share", "touches_per_game",
        "rush_att_per_game", "targets_per_game", "td_probability_raw",
        "td_probability", "fair_american_odds",
    ]
    cols = [c for c in keep if c in scored.columns]
    out = scored[cols].copy()
    out["td_probability_pct"] = 100 * out["td_probability"]
    out = out.sort_values(
        ["td_probability", "baseline_score"], ascending=[False, False]
    ).reset_index(drop=True)
    out.insert(0, "rank", np.arange(1, len(out) + 1))
    return out


def main():
    args = parse_args()
    data_path = Path(args.data)
    model_path = Path(args.model)

    if not data_path.exists():
        raise SystemExit(f"Dataset not found: {data_path}")
    if not model_path.exists():
        raise SystemExit(f"Model not found: {model_path}\nRun train_historical.py first.")

    print(f"Loading historical features: {data_path}")
    df = pd.read_parquet(data_path)
    print(f"Loaded {len(df):,} player-games")

    print(f"Loading trained model: {model_path}")
    model = joblib.load(model_path)

    existing = _find_existing_week(df, args.season, args.week)
    if not existing.empty:
        mode = "historical-existing-week"
        print(
            f"Found {len(existing):,} existing rows for {args.season} Week {args.week}; "
            "scoring their leakage-safe pregame features."
        )
        rows = existing
    else:
        mode = "upcoming-role-only"
        print(
            f"No completed rows exist for {args.season} Week {args.week}.\n"
            "Building upcoming-week rows from each player's completed history..."
        )
        rows = _build_upcoming_rows(df, args.season, args.week)
        print(f"Built {len(rows):,} upcoming player rows")

    if rows.empty:
        raise SystemExit("No players available to score.")

    rankings = _score(model, rows)
    rankings["scoring_mode"] = mode

    output = Path(args.output) if args.output else Path(
        f"data/output/{args.season}_week_{args.week}_td_rankings.csv"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    rankings.to_csv(output, index=False)

    print("\nIMPORTANT: This V2 weekly scorer is ROLE-ONLY.")
    print("Opponent matchup, Vegas implied points, injuries, and projected game environment are not in the model yet.\n")

    display_cols = [
        c for c in [
            "rank", "player_name", "position", "posteam",
            "baseline_score", "td_probability_pct", "fair_american_odds",
            "touches_per_game", "goal_line_share",
        ] if c in rankings.columns
    ]
    printable = rankings[display_cols].head(args.top).copy()
    for c in ["baseline_score", "td_probability_pct", "touches_per_game", "goal_line_share"]:
        if c in printable.columns:
            printable[c] = printable[c].round(2)

    print(printable.to_string(index=False))
    print(f"\nSaved {len(rankings):,} rankings to {output}")


if __name__ == "__main__":
    main()
