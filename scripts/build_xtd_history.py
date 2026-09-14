from pathlib import Path
import argparse
import sys
import joblib

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from td_engine.io import load_nflverse_history
from td_engine.x_td import ExpectedTDModel, build_xtd_opportunities


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", type=int, default=2021)
    ap.add_argument("--end", type=int, default=2026)
    ap.add_argument("--train-through", type=int, default=2025)
    args = ap.parse_args()

    proc = ROOT / "data" / "processed"
    proc.mkdir(parents=True, exist_ok=True)

    seasons = list(range(args.start, args.end + 1))
    print(f"Loading PBP for {seasons}...", flush=True)
    history = load_nflverse_history(
        seasons,
        include_player_stats=False,
        include_schedules=False,
    )
    pbp = history["pbp"]
    print(f"Loaded {len(pbp):,} PBP rows", flush=True)

    opps = build_xtd_opportunities(pbp)
    print(f"Built {len(opps):,} rush/target opportunities", flush=True)

    train = opps[opps["season"] <= args.train_through].copy()
    print(f"Training xTD through {args.train_through} on {len(train):,} opportunities", flush=True)
    model = ExpectedTDModel().fit(train)

    model_path = proc / "xtd_model.joblib"
    joblib.dump(model, model_path)

    pred = model.predict_opportunities(opps)
    opp_path = proc / "xtd_opportunities.parquet"
    pred.to_parquet(opp_path, index=False)

    group_cols = [c for c in ["season", "week", "game_id", "posteam", "player_id", "player_name"] if c in pred.columns]
    agg = (
        pred.groupby(group_cols, as_index=False)
        .agg(
            expected_tds=("xTD", "sum"),
            actual_tds=("touchdown", "sum"),
            xtd_opps=("xTD", "size"),
        )
    )
    pg_path = proc / "xtd_player_games.parquet"
    agg.to_parquet(pg_path, index=False)

    print(f"Saved xTD model: {model_path}")
    print(f"Saved xTD opportunities: {opp_path}")
    print(f"Saved xTD player-games: {pg_path}")
    print(f"Historical xTD total: {agg['expected_tds'].sum():.1f}")
    print(f"Historical actual TD total: {agg['actual_tds'].sum():.0f}")


if __name__ == "__main__":
    main()
