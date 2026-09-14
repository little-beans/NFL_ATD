import pandas as pd
from td_engine.matchup import build_defense_pregame_features
from td_engine.schedule import schedule_to_team_rows

def test_defense_shift_no_same_game_leakage():
    rows=[]
    for week,tds in [(1,1),(2,3),(3,0)]:
        rows.append(dict(season=2025,week=week,game_id=f'g{week}',posteam='A',defteam='B',player_id=f'p{week}',td=tds,team_rz_opp=4,team_inside10_opp=3,team_inside5_opp=2,position='RB',season_type='REG'))
    d=pd.DataFrame(rows)
    f=build_defense_pregame_features(d)
    w2=f[f.week.eq(2)].iloc[0]
    assert abs(w2.def_td_allowed_pg-1.0)<1e-9

def test_schedule_team_rows():
    s=pd.DataFrame([dict(season=2026,week=2,game_id='g',home_team='KC',away_team='DEN',spread_line=3.0,total_line=47.0,home_rest=7,away_rest=7,game_type='REG')])
    x=schedule_to_team_rows(s)
    kc=x[x.posteam.eq('KC')].iloc[0]; den=x[x.posteam.eq('DEN')].iloc[0]
    assert kc.defteam=='DEN' and den.defteam=='KC'
    assert kc.implied_team_points==25.0 and den.implied_team_points==22.0
