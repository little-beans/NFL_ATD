from pathlib import Path
import sys, argparse, joblib
ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))
import pandas as pd
from td_engine.model import TDProbabilityModel

FEATURES=[
 'red_zone_share','volume_share','goal_line_share','target_share','td_efficiency_score',
 'def_td_allowed_pg','def_rz_td_rate','def_inside10_td_rate','def_inside5_td_rate',
 'def_rb_td_allowed_pg','def_wr_td_allowed_pg','def_te_td_allowed_pg',
 'implied_team_points','total_line','team_spread','is_home'
]

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--data', default='data/processed/td_matchup_features.parquet')
    ap.add_argument('--train-through', type=int, default=2024)
    ap.add_argument('--calibrate', type=int, default=2025)
    ap.add_argument('--test', type=int, default=2026)
    ap.add_argument('--output', default='data/processed/td_matchup_model.joblib')
    args=ap.parse_args()
    df=pd.read_parquet(args.data)
    if 'season_type' in df.columns: df=df[df['season_type'].fillna('REG').eq('REG')].copy()
    # Need at least one prior game worth of player history.
    if 'career_games_before' in df.columns: df=df[df['career_games_before'].fillna(0)>=1].copy()
    for c in FEATURES:
        if c not in df.columns: raise SystemExit(f'Missing feature {c}; run build_matchup_history.py')
    # Stable neutral fill for defense fields; market fields already filled by builder.
    for c in FEATURES:
        if df[c].isna().any(): df[c]=df[c].fillna(df[c].median())

    train=df[df['season']<=args.train_through].copy()
    cal=df[df['season'].eq(args.calibrate)].copy()
    test=df[df['season'].eq(args.test)].copy()
    print(f'Train: {len(train):,}; calibration: {len(cal):,}; test: {len(test):,}')
    m=TDProbabilityModel(features=FEATURES)
    m.fit_base(train); m.fit_calibrator(cal)
    print('Calibration metrics:', m.evaluate(cal))
    print('Test metrics:', m.evaluate(test))
    out=Path(args.output); out.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(m,out)
    print(f'Saved matchup model to {out}')

if __name__=='__main__': main()
