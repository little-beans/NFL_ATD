from __future__ import annotations

import argparse
import sys
from pathlib import Path
import time

# Allow `python scripts/build_history.py` from the project root on Windows/macOS/Linux.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from td_engine.io import load_nflverse_history
from td_engine.dataset import build_training_dataset, save_dataset


def parse_args():
    p = argparse.ArgumentParser(description="Download nflverse history and build leakage-safe TD training data.")
    p.add_argument("--start", type=int, default=2021)
    p.add_argument("--end", type=int, default=2026)
    p.add_argument("--cache-dir", default="data/raw")
    p.add_argument("--output", default="data/processed/td_pregame_features.parquet")
    p.add_argument("--force", action="store_true")
    p.add_argument("--min-prior-games", type=int, default=1)
    p.add_argument("--skip-player-stats", action="store_true", help="Build from PBP only (faster, but no position metadata).")
    return p.parse_args()


def main():
    args = parse_args()
    seasons = list(range(args.start, args.end + 1))

    print("NFL Anytime TD historical builder", flush=True)
    print(f"Project root: {PROJECT_ROOT}", flush=True)
    print(f"Seasons: {seasons}", flush=True)
    print("Downloading one season at a time so progress stays visible.\n", flush=True)

    total_start = time.perf_counter()
    data = load_nflverse_history(
        seasons,
        cache_dir=args.cache_dir,
        force=args.force,
        include_player_stats=not args.skip_player_stats,
        include_schedules=False,
        verbose=True,
    )

    print("\n=== Building player-game + leakage-safe rolling features ===", flush=True)
    t0 = time.perf_counter()
    ds = build_training_dataset(
        data["pbp"],
        player_stats=data.get("player_stats"),
        min_prior_games=args.min_prior_games,
    )
    print(f"Feature build completed in {time.perf_counter()-t0:.1f}s", flush=True)

    print("Saving final dataset...", flush=True)
    path = save_dataset(ds, args.output)
    print(f"Saved {len(ds):,} player-games to {path}", flush=True)
    if "season" in ds.columns:
        print(f"Seasons: {sorted(ds.season.dropna().astype(int).unique().tolist())}", flush=True)
    print(f"TD rate: {ds.scored_td.mean():.3%}", flush=True)
    print(f"Total runtime: {time.perf_counter()-total_start:.1f}s", flush=True)


if __name__ == "__main__":
    main()
