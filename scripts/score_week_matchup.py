from pathlib import Path
import sys, argparse, joblib
ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))
# script directory is also on sys.path when executed directly
import pandas as pd
import numpy as np
from score_week import _build_upcoming_rows
from td_engine.matchup import latest_defense_priors
from td_engine.schedule import load_week_team_schedule


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--season',type=int,required=True)
    ap.add_argument('--week',type=int,required=True)
    ap.add_argument('--data',default='data/processed/td_pregame_features.parquet')
    ap.add_argument('--model',default='data/processed/td_matchup_model.joblib')
    ap.add_argument('--output',default=None)
    ap.add_argument('--top',type=int,default=40)
    args=ap.parse_args()

    df=pd.read_parquet(args.data)
    model=joblib.load(args.model)
    print(f'Loaded {len(df):,} historical player-games')
    print('Building role priors...', flush=True)
    rows=_build_upcoming_rows(df,args.season,args.week)

    print('Loading target-week schedule and market lines...', flush=True)
    sched=load_week_team_schedule(args.season,args.week)
    if sched.empty: raise SystemExit(f'No schedule rows found for {args.season} Week {args.week}')
    print(f'Found {len(sched)//2} games')
    sched_cols=['posteam','defteam','game_id','is_home','total_line','team_spread','implied_team_points','rest_days','gameday','gametime']
    if 'posteam' not in rows.columns:
        raise SystemExit(
            "Upcoming player rows are missing 'posteam'. Rebuild the historical dataset "
            "with team metadata before scoring."
        )
    # Preserve the join key (`posteam`) and replace only schedule-derived fields.
    replace_cols=[c for c in sched_cols if c != 'posteam' and c in rows.columns]
    rows=rows.drop(columns=replace_cols, errors='ignore')
    schedule_view=sched[[c for c in sched_cols if c in sched.columns]].drop_duplicates('posteam')
    rows=rows.merge(schedule_view, on='posteam', how='inner', validate='many_to_one')

    print('Building opponent defensive priors...', flush=True)
    defense=latest_defense_priors(df,args.season,args.week)
    rows=rows.merge(defense,on='defteam',how='left')

    # Neutral fallbacks.
    for c in model.features:
        if c not in rows.columns: raise SystemExit(f'Missing model feature: {c}')
        if rows[c].isna().any():
            if c=='total_line': rows[c]=rows[c].fillna(44.0)
            elif c in ('team_spread',): rows[c]=rows[c].fillna(0.0)
            elif c=='implied_team_points': rows[c]=rows[c].fillna(rows['total_line']/2.0)
            else: rows[c]=rows[c].fillna(rows[c].median())

    scored=model.predict(rows)
    scored['td_probability_pct']=100*scored['td_probability']
    scored=scored.sort_values(['td_probability','baseline_score'],ascending=False).reset_index(drop=True)
    scored.insert(0,'rank',np.arange(1,len(scored)+1))
    keep=['rank','player_name','position','posteam','defteam','is_home','baseline_score','td_probability_pct','fair_american_odds','implied_team_points','total_line','team_spread','def_td_allowed_pg','def_rz_td_rate','goal_line_share','red_zone_share','touches_per_game','targets_per_game']
    out=scored[[c for c in keep if c in scored.columns]].copy()
    path=Path(args.output) if args.output else Path(f'data/output/{args.season}_week_{args.week}_td_matchup_rankings.csv')
    path.parent.mkdir(parents=True, exist_ok=True); out.to_csv(path,index=False)
    show=out.head(args.top).copy()
    for c in ['baseline_score','td_probability_pct','implied_team_points','def_td_allowed_pg','def_rz_td_rate','goal_line_share']:
        if c in show.columns: show[c]=show[c].round(2)
    print('\nMATCHUP-AWARE ANYTIME TD RANKINGS\n')
    print(show.to_string(index=False))
    print(f'\nSaved {len(out):,} rankings to {path}')

if __name__=='__main__': main()
