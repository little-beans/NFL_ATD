import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from td_engine.odds_value import NFL_TEAM_NAMES, extract_atd_prices


API_BASE = "https://api.the-odds-api.com/v4"
SPORT = "americanfootball_nfl"
MARKET = "player_anytime_td"


def _load_env():
    try:
        from dotenv import load_dotenv
        load_dotenv(ROOT / ".env")
        load_dotenv()
    except Exception:
        pass


def _read_rankings(season: int, week: int) -> pd.DataFrame:
    path = ROOT/"data"/"output"/f"{season}_week_{week}_td_unified_rankings.csv"
    if not path.exists():
        raise FileNotFoundError(
            f"Missing unified rankings: {path}\n"
            "Run score_week_unified.py first."
        )
    return pd.read_csv(path)


def _cache_fresh(path: Path, minutes: int) -> bool:
    if not path.exists():
        return False
    age_seconds = datetime.now(timezone.utc).timestamp() - path.stat().st_mtime
    return age_seconds <= minutes * 60


def _quota(headers):
    return {
        "remaining": headers.get("x-requests-remaining"),
        "used": headers.get("x-requests-used"),
        "last": headers.get("x-requests-last"),
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--season", type=int, required=True)
    p.add_argument("--week", type=int, required=True)
    p.add_argument("--region", default="us")
    p.add_argument("--bookmakers", default="")
    p.add_argument("--cache-minutes", type=int, default=15)
    p.add_argument("--refresh", action="store_true")
    p.add_argument(
        "--max-events",
        type=int,
        default=0,
        help="Optional safety cap. 0 means all matching events.",
    )
    args = p.parse_args()

    _load_env()

    api_key = os.getenv("ODDS_API_KEY")
    if not api_key:
        raise RuntimeError(
            "ODDS_API_KEY is not set. Put it in your local .env file:\n"
            "ODDS_API_KEY=your_key_here"
        )

    rankings = _read_rankings(args.season, args.week)
    target_abbrs = sorted(set(rankings["posteam"].dropna().astype(str)))
    target_full = {
        NFL_TEAM_NAMES[t]
        for t in target_abbrs
        if t in NFL_TEAM_NAMES
    }

    session = requests.Session()
    session.headers.update({"User-Agent": "NFL-ATD-Value-Engine/1.0"})

    print("Loading NFL events (free endpoint)...")
    r = session.get(
        f"{API_BASE}/sports/{SPORT}/events",
        params={"apiKey": api_key, "dateFormat": "iso"},
        timeout=30,
    )
    r.raise_for_status()
    events = r.json()

    relevant = [
        e for e in events
        if e.get("home_team") in target_full
        or e.get("away_team") in target_full
    ]

    relevant = sorted(
        relevant,
        key=lambda e: e.get("commence_time", ""),
    )

    if args.max_events > 0:
        relevant = relevant[:args.max_events]

    print(f"Relevant upcoming events: {len(relevant)}")

    cache_dir = (
        ROOT/"data"/"cache"/"odds"/
        f"{args.season}_week_{args.week}"/MARKET
    )
    cache_dir.mkdir(parents=True, exist_ok=True)

    all_prices = []
    api_calls = 0
    cache_hits = 0
    last_quota = {}

    for i, event in enumerate(relevant, start=1):
        event_id = event["id"]
        home = event.get("home_team")
        away = event.get("away_team")

        cache_path = cache_dir/f"{event_id}.json"

        if not args.refresh and _cache_fresh(cache_path, args.cache_minutes):
            payload = json.loads(cache_path.read_text(encoding="utf-8"))
            cache_hits += 1
            source = "cache"
        else:
            params = {
                "apiKey": api_key,
                "regions": args.region,
                "markets": MARKET,
                "oddsFormat": "american",
                "dateFormat": "iso",
            }
            if args.bookmakers.strip():
                params["bookmakers"] = args.bookmakers.strip()

            print(f"[{i}/{len(relevant)}] Fetching {away} @ {home}...")
            resp = session.get(
                f"{API_BASE}/sports/{SPORT}/events/{event_id}/odds",
                params=params,
                timeout=30,
            )
            resp.raise_for_status()
            payload = resp.json()
            cache_path.write_text(
                json.dumps(payload, indent=2),
                encoding="utf-8",
            )
            api_calls += 1
            last_quota = _quota(resp.headers)
            source = "api"

        prices = extract_atd_prices(payload)
        if not prices.empty:
            prices["fetch_source"] = source
            all_prices.append(prices)

    if all_prices:
        odds = pd.concat(all_prices, ignore_index=True)
    else:
        odds = pd.DataFrame()

    out = (
        ROOT/"data"/"processed"/
        f"atd_odds_{args.season}_week_{args.week}.parquet"
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    odds.to_parquet(out, index=False)

    print()
    print(f"ATD price rows: {len(odds):,}")
    print(f"API calls made: {api_calls}")
    print(f"Cache hits: {cache_hits}")

    if last_quota:
        print(
            "API quota — "
            f"last call cost: {last_quota.get('last')}, "
            f"used: {last_quota.get('used')}, "
            f"remaining: {last_quota.get('remaining')}"
        )

    print(f"Saved odds: {out}")


if __name__ == "__main__":
    main()
