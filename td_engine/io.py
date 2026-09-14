from __future__ import annotations

from pathlib import Path
from typing import Iterable
import time
import pandas as pd


def _to_pandas(frame):
    """Convert a Polars/pandas dataframe to pandas without taking a hard Polars dependency."""
    if isinstance(frame, pd.DataFrame):
        return frame.copy()
    if hasattr(frame, "to_pandas"):
        return frame.to_pandas()
    raise TypeError(f"Unsupported dataframe type: {type(frame)!r}")


def _load_one_season(name, season, loader, cache_dir: Path, force=False, verbose=True):
    path = cache_dir / f"{name}_{season}.parquet"
    if path.exists() and not force:
        if verbose:
            print(f"[{name}] {season}: loading cached {path}", flush=True)
        t0 = time.perf_counter()
        df = pd.read_parquet(path)
        if verbose:
            print(f"[{name}] {season}: cache loaded ({len(df):,} rows) in {time.perf_counter()-t0:.1f}s", flush=True)
        return df

    if verbose:
        print(f"[{name}] {season}: downloading...", flush=True)
    t0 = time.perf_counter()
    frame = loader(season)
    if verbose:
        print(f"[{name}] {season}: download returned in {time.perf_counter()-t0:.1f}s; converting to pandas...", flush=True)
    df = _to_pandas(frame)
    if verbose:
        print(f"[{name}] {season}: converted ({len(df):,} rows); caching...", flush=True)
    df.to_parquet(path, index=False)
    if verbose:
        print(f"[{name}] {season}: saved {path} in {time.perf_counter()-t0:.1f}s total", flush=True)
    return df


def load_nflverse_history(
    seasons: Iterable[int],
    cache_dir: str | Path = "data/raw",
    force: bool = False,
    include_player_stats: bool = True,
    include_schedules: bool = False,
    verbose: bool = True,
) -> dict[str, pd.DataFrame]:
    """
    Download/cache nflverse data one season at a time.

    Season-by-season loading is intentionally used because multi-season PBP loads can
    appear frozen for a long time on Windows and use much more memory.
    """
    seasons = sorted({int(s) for s in seasons})
    if not seasons:
        raise ValueError("At least one season is required.")

    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    try:
        import nflreadpy as nfl
    except ImportError as exc:
        raise ImportError(
            "nflreadpy is required. Install dependencies with `pip install -r requirements.txt`."
        ) from exc

    out: dict[str, pd.DataFrame] = {}

    if verbose:
        print(f"Loading nflverse seasons: {seasons}", flush=True)
        print(f"Cache directory: {cache_dir.resolve()}", flush=True)

    pbp_parts = []
    for i, season in enumerate(seasons, 1):
        if verbose:
            print(f"\n=== PBP season {season} ({i}/{len(seasons)}) ===", flush=True)
        pbp_parts.append(
            _load_one_season(
                "pbp", season,
                lambda s: nfl.load_pbp([s]),
                cache_dir, force=force, verbose=verbose,
            )
        )
    out["pbp"] = pd.concat(pbp_parts, ignore_index=True, sort=False)
    if verbose:
        print(f"\nPBP combined: {len(out['pbp']):,} rows", flush=True)

    if include_player_stats:
        stats_parts = []
        for i, season in enumerate(seasons, 1):
            if verbose:
                print(f"\n=== Player stats {season} ({i}/{len(seasons)}) ===", flush=True)
            stats_parts.append(
                _load_one_season(
                    "player_stats", season,
                    lambda s: nfl.load_player_stats([s], summary_level="week"),
                    cache_dir, force=force, verbose=verbose,
                )
            )
        out["player_stats"] = pd.concat(stats_parts, ignore_index=True, sort=False)
        if verbose:
            print(f"Player stats combined: {len(out['player_stats']):,} rows", flush=True)

    if include_schedules:
        sched_parts = []
        for i, season in enumerate(seasons, 1):
            if verbose:
                print(f"\n=== Schedules {season} ({i}/{len(seasons)}) ===", flush=True)
            sched_parts.append(
                _load_one_season(
                    "schedules", season,
                    lambda s: nfl.load_schedules([s]),
                    cache_dir, force=force, verbose=verbose,
                )
            )
        out["schedules"] = pd.concat(sched_parts, ignore_index=True, sort=False)

    return out
