import argparse, json, os, sys, time
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


def _read_rankings(season, week):
    p = ROOT/"data"/"output"/f"{season}_week_{week}_td_unified_rankings.csv"
    if not p.exists():
        raise FileNotFoundError(p)
    return pd.read_csv(p)


def _cache_fresh(path, minutes):
    if not path.exists():
        return False
    age = datetime.now(timezone.utc).timestamp() - path.stat().st_mtime
    return age <= minutes * 60


def _quota(headers):
    return {
        "remaining": headers.get("x-requests-remaining"),
        "used": headers.get("x-requests-used"),
        "last": headers.get("x-requests-last"),
    }


def _get_with_retry(session, url, params, max_retries=5):
    last = None
    for attempt in range(max_retries + 1):
        r = session.get(url, params=params, timeout=30)
        last = r
        if r.status_code == 429:
            if attempt >= max_retries:
                break
            retry_after = r.headers.get("Retry-After")
            try:
                delay = float(retry_after) if retry_after else 2 * (2 ** attempt)
            except Exception:
                delay = 2 * (2 ** attempt)
            delay = min(delay, 30)
            print(f"  429 rate limit. Retrying in {delay:.1f}s...")
            time.sleep(delay)
            continue
        if 500 <= r.status_code < 600:
            if attempt >= max_retries:
                break
            delay = min(2 * (2 ** attempt), 30)
            print(f"  Server error {r.status_code}. Retrying in {delay:.1f}s...")
            time.sleep(delay)
            continue
        r.raise_for_status()
        return r
    last.raise_for_status()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--season", type=int, required=True)
    ap.add_argument("--week", type=int, required=True)
    ap.add_argument("--region", default="us")
    ap.add_argument("--bookmakers", default="")
    ap.add_argument("--cache-minutes", type=int, default=15)
    ap.add_argument("--refresh", action="store_true")
    ap.add_argument("--max-events", type=int, default=0)
    ap.add_argument("--request-delay", type=float, default=1.25)
    ap.add_argument("--max-retries", type=int, default=5)
    args = ap.parse_args()

    _load_env()
    api_key = os.getenv("ODDS_API_KEY")
    if not api_key:
        raise RuntimeError("ODDS_API_KEY missing from .env")

    rankings = _read_rankings(args.season, args.week)
    target_full = {
        NFL_TEAM_NAMES[t] for t in set(rankings["posteam"].dropna().astype(str))
        if t in NFL_TEAM_NAMES
    }

    session = requests.Session()
    session.headers.update({"User-Agent": "NFL-ATD-Value-Engine/1.0"})

    print("Loading NFL events (free endpoint)...")
    er = _get_with_retry(
        session,
        f"{API_BASE}/sports/{SPORT}/events",
        {"apiKey": api_key, "dateFormat": "iso"},
        args.max_retries,
    )
    events = er.json()

    relevant = sorted(
        [e for e in events if e.get("home_team") in target_full or e.get("away_team") in target_full],
        key=lambda e: e.get("commence_time", "")
    )
    if args.max_events > 0:
        relevant = relevant[:args.max_events]

    print(f"Relevant upcoming events: {len(relevant)}")

    cache_dir = ROOT/"data"/"cache"/"odds"/f"{args.season}_week_{args.week}"/MARKET
    cache_dir.mkdir(parents=True, exist_ok=True)

    all_prices, failed = [], []
    api_calls = cache_hits = 0
    last_quota = {}

    for i, event in enumerate(relevant, 1):
        event_id = event["id"]
        away, home = event.get("away_team"), event.get("home_team")
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
            try:
                resp = _get_with_retry(
                    session,
                    f"{API_BASE}/sports/{SPORT}/events/{event_id}/odds",
                    params,
                    args.max_retries,
                )
                payload = resp.json()
                cache_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
                api_calls += 1
                last_quota = _quota(resp.headers)
                source = "api"
            except requests.HTTPError as exc:
                print(f"  Failed after retries: {exc}")
                failed.append((away, home, event_id))
                if cache_path.exists():
                    print("  Using previous cached event odds.")
                    payload = json.loads(cache_path.read_text(encoding="utf-8"))
                    cache_hits += 1
                    source = "stale_cache"
                else:
                    continue

            if args.request_delay > 0 and i < len(relevant):
                time.sleep(args.request_delay)

        prices = extract_atd_prices(payload)
        if not prices.empty:
            prices["fetch_source"] = source
            all_prices.append(prices)

    odds = pd.concat(all_prices, ignore_index=True) if all_prices else pd.DataFrame()

    out = ROOT/"data"/"processed"/f"atd_odds_{args.season}_week_{args.week}.parquet"
    out.parent.mkdir(parents=True, exist_ok=True)

    if odds.empty:
        raise RuntimeError("No ATD odds returned; existing odds file was not replaced.")

    tmp = out.with_suffix(".tmp.parquet")
    odds.to_parquet(tmp, index=False)
    tmp.replace(out)

    print()
    print(f"ATD price rows: {len(odds):,}")
    print(f"API calls made: {api_calls}")
    print(f"Cache hits: {cache_hits}")
    print(f"Failed event refreshes: {len(failed)}")
    if last_quota:
        print(
            f"API quota — last call cost: {last_quota['last']}, "
            f"used: {last_quota['used']}, remaining: {last_quota['remaining']}"
        )
    print(f"Saved odds: {out}")


if __name__ == "__main__":
    main()
