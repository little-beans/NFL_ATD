from pathlib import Path
import sys, argparse
ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))
import pandas as pd
from td_engine.matchup import build_defense_pregame_features
from td_engine.schedule import load_schedules_pandas, schedule_to_team_rows


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--data', default='data/processed/td_pregame_features.parquet')
    ap.add_argument('--output', default='data/processed/td_matchup_features.parquet')
    args=ap.parse_args()
    p=Path(args.data)
    if not p.exists(): raise SystemExit(f'Missing {p}. Run build_history.py first.')
    print(f'Loading {p}...', flush=True)
    df=pd.read_parquet(p)
    print(f'Loaded {len(df):,} player-games', flush=True)

    print('Building leakage-safe opponent defense features...', flush=True)
    defense=build_defense_pregame_features(df)
    out=df.merge(defense, on=['season','week','game_id','posteam','defteam'], how='left')

    seasons=sorted(out['season'].dropna().astype(int).unique().tolist())
    print(f'Loading schedules/market lines for {seasons[0]}-{seasons[-1]}...', flush=True)
    sched=schedule_to_team_rows(load_schedules_pandas(seasons))
    market_cols=['season','week','game_id','posteam','defteam','is_home','total_line','team_spread','implied_team_points','rest_days']
    market=sched[[c for c in market_cols if c in sched.columns]].drop_duplicates(['season','week','game_id','posteam'])
    out=out.merge(market, on=['season','week','game_id','posteam'], how='left', suffixes=('','_sched'))
    if 'defteam_sched' in out.columns:
        out['defteam']=out['defteam'].fillna(out['defteam_sched'])
        out=out.drop(columns=['defteam_sched'])

    # Conservative neutral fills for games without available lines.
    out['total_line']=out['total_line'].fillna(44.0)
    out['team_spread']=out['team_spread'].fillna(0.0)
    out['implied_team_points']=out['implied_team_points'].fillna(out['total_line']/2.0)
    for c in ['is_home']:
        if c in out.columns: out[c]=out[c].fillna(0)

    op=Path(args.output); op.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(op, index=False)
    print(f'Saved {len(out):,} rows to {op}', flush=True)
    print('Matchup columns:', [c for c in out.columns if c.startswith('def_') or c in ['total_line','team_spread','implied_team_points','is_home']])

if __name__=='__main__': main()
