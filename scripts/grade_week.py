
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

def profit_per_unit(odds):
    odds=float(odds)
    return odds/100.0 if odds>0 else 100.0/abs(odds)

def classify(row):
    edge=float(row.get("edge_probability_points", -999))
    roi=float(row.get("expected_roi_pct", -999))
    books=int(row.get("books_available", 0))
    role=float(row.get("role_support_score", 0))
    disp=float(row.get("market_dispersion_pp", 0))
    base=roi>0 and role>=35 and disp<=7.5
    if base and edge>=7 and books>=3: return "STRONG VALUE"
    if base and 4<=edge<7 and books>=3: return "ACTIONABLE"
    if base and 2<=edge<4 and books>=2: return "WATCHLIST"
    if edge>=2 and roi>0: return "REVIEW ONLY"
    return "NO EDGE"

def load_actuals(season, week):
    p=ROOT/"data"/"processed"/"xtd_player_games.parquet"
    if not p.exists():
        raise FileNotFoundError(f"{p}\nRebuild xTD history first.")
    d=pd.read_parquet(p)
    actual_col=next((c for c in ["actual_tds","touchdowns","total_tds","scored_td"] if c in d.columns),None)
    if actual_col is None:
        raise KeyError("No actual TD column found in xtd_player_games.parquet")
    w=d[(pd.to_numeric(d["season"],errors="coerce")==season)&(pd.to_numeric(d["week"],errors="coerce")==week)].copy()
    if w.empty:
        raise RuntimeError(f"No actual TD rows found for {season} Week {week}. Rebuild xTD history.")
    a=w.groupby("player_id",as_index=False)[actual_col].sum().rename(columns={actual_col:"actual_tds"})
    a["scored_td"]=a["actual_tds"]>0
    return a

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--season",type=int,required=True)
    ap.add_argument("--week",type=int,required=True)
    args=ap.parse_args()

    board_path=ROOT/"data"/"output"/f"{args.season}_week_{args.week}_atd_value_board.csv"
    board=pd.read_csv(board_path)
    actuals=load_actuals(args.season,args.week)

    board["tier"]=board.apply(classify,axis=1)
    g=board.merge(actuals,on="player_id",how="left")
    g["actual_tds"]=g["actual_tds"].fillna(0)
    g["scored_td"]=g["scored_td"].fillna(False)
    g["result"]=np.where(g["scored_td"],"HIT","MISS")
    g["flat_1u_pnl"]=[profit_per_unit(o) if hit else -1.0 for hit,o in zip(g["scored_td"],g["sportsbook_odds"])]

    rows=[]
    for tier in ["STRONG VALUE","ACTIONABLE","WATCHLIST","REVIEW ONLY"]:
        x=g[g["tier"]==tier]
        n=len(x); hits=int(x["scored_td"].sum()) if n else 0
        units=float(x["flat_1u_pnl"].sum()) if n else 0.0
        rows.append({
            "tier":tier,"bets":n,"hits":hits,
            "hit_rate_pct":(100*hits/n if n else np.nan),
            "units":units,"roi_pct":(100*units/n if n else np.nan)
        })
    s=pd.DataFrame(rows)

    out=ROOT/"data"/"output"
    gp=out/f"{args.season}_week_{args.week}_atd_results.csv"
    sp=out/f"{args.season}_week_{args.week}_atd_results_summary.csv"
    g.to_csv(gp,index=False); s.to_csv(sp,index=False)

    print("="*72)
    print(f"NFL ATD RESULTS — {args.season} WEEK {args.week}")
    print("="*72)
    print(s.to_string(index=False))
    print()
    cols=[c for c in ["player_name","posteam","tier","sportsbook_odds","edge_probability_points","actual_tds","result","flat_1u_pnl"] if c in g.columns]
    print(g[g["tier"]!="NO EDGE"][cols].to_string(index=False))
    print()
    print(f"Saved: {gp}")
    print(f"Saved: {sp}")

if __name__=="__main__":
    main()
