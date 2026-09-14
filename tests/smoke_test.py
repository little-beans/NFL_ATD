import pandas as pd
from td_engine.rolling import build_rolling_pregame_features
from td_engine.score import baseline_score

rows=[]
for pid, vals in [("A", [10,12,14,16]), ("B", [4,7,9,6])]:
    for i,t in enumerate(vals,1):
        rows.append({
            "player_id":pid,"player_name":pid,"season":2025,"week":i,"game_id":f"{pid}{i}","season_type":"REG",
            "touches":t,"rush_att":max(t-3,0),"targets":3,"receptions":2,"td":1 if i==2 else 0,
            "rz_opp":2,"inside10_opp":1,"inside5_opp":1,"rz_carry":1,"inside10_carry":1,"inside5_carry":1,
            "rz_target":1,"inside10_target":0,"inside5_target":0,
            "red_zone_share_game":.25,"inside10_share_game":.25,"goal_line_share_game":.5,
            "volume_share_game":.3,"carry_share_game":.3,"target_share_game":.15,
            "rz_carry_share_game":.3,"inside5_carry_share_game":.5,"rz_target_share_game":.2,
            "scored_td":1 if i==2 else 0,
        })

df=pd.DataFrame(rows)
out=baseline_score(build_rolling_pregame_features(df))
assert out.loc[(out.player_id=="A") & (out.week==2), "pregame_touches_l3"].iloc[0] == 10
assert len(out)==8
print(out[["player_id","week","career_games_before","red_zone_share","volume_share","baseline_score"]].to_string(index=False))
print("SMOKE TEST PASSED")
