from __future__ import annotations

import argparse
import html
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def _num(value, default=None):
    x = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    if pd.isna(x):
        return default
    return float(x)


def _fmt_pct_decimal(value):
    x = _num(value)
    return "-" if x is None else f"{100 * x:.1f}%"


def _fmt_pct(value):
    x = _num(value)
    return "-" if x is None else f"{x:.1f}%"


def _fmt_pp(value):
    x = _num(value)
    return "-" if x is None else f"{x:+.1f} pp"


def _fmt_odds(value):
    x = _num(value)
    if x is None:
        return "-"
    n = int(round(x))
    return f"+{n}" if n > 0 else str(n)


def _txt(value, fallback="-"):
    if value is None:
        return fallback
    s = str(value).strip()
    if not s or s.lower() == "nan":
        return fallback
    return s


def classify_tier(row):
    edge = _num(row.get("edge_probability_points"), -999)
    roi = _num(row.get("expected_roi_pct"), -999)
    books = int(_num(row.get("books_available"), 0) or 0)
    role = _num(row.get("role_support_score"), 0)
    dispersion = _num(row.get("market_dispersion_pp"), 0)

    base_ok = roi > 0 and role >= 35 and dispersion <= 7.5

    if base_ok and edge >= 7 and books >= 3:
        return "STRONG VALUE"
    if base_ok and 4 <= edge < 7 and books >= 3:
        return "ACTIONABLE"
    if base_ok and 2 <= edge < 4 and books >= 2:
        return "WATCHLIST"
    if edge >= 2 and roi > 0:
        return "REVIEW ONLY"
    return "NO EDGE"


def row_to_record(row):
    return {
        "player": _txt(row.get("player_name")),
        "team": _txt(row.get("posteam")),
        "position": _txt(row.get("position")),
        "model_probability": _fmt_pct_decimal(row.get("model_probability")),
        "model_fair_odds": _fmt_odds(row.get("model_fair_odds")),
        "book": _txt(row.get("book_title")),
        "sportsbook_odds": _fmt_odds(row.get("sportsbook_odds")),
        "market_probability": _fmt_pct_decimal(row.get("sportsbook_implied_probability")),
        "edge": _fmt_pp(row.get("edge_probability_points")),
        "roi": _fmt_pct(row.get("expected_roi_pct")),
        "books": int(_num(row.get("books_available"), 0) or 0),
        "role": "-" if _num(row.get("role_support_score")) is None else f"{_num(row.get('role_support_score')):.0f}",
        "xtd5": "-" if _num(row.get("expected_tds_l5")) is None else f"{_num(row.get('expected_tds_l5')):.2f}",
        "tag": _txt(row.get("td_debt_tag"), ""),
        "flags": _txt(row.get("quality_flags"), ""),
        "tier": classify_tier(row),
        "edge_num": _num(row.get("edge_probability_points"), -999),
        "roi_num": _num(row.get("expected_roi_pct"), -999),
    }


def render_card(record):
    tag_html = ""
    if record["tag"]:
        tag_html = f'<span class="tag">{html.escape(record["tag"])}</span>'

    flags_html = ""
    if record["flags"]:
        flags = html.escape(record["flags"]).replace("|", " • ")
        flags_html = f'<div class="flags">{flags}</div>'

    return f"""
    <article class="player-card">
      <div class="card-top">
        <div>
          <div class="player">{html.escape(record["player"])}</div>
          <div class="meta">{html.escape(record["team"])} · {html.escape(record["position"])} {tag_html}</div>
        </div>
        <div class="best-price">
          <div class="price">{html.escape(record["sportsbook_odds"])}</div>
          <div class="book">{html.escape(record["book"])}</div>
        </div>
      </div>

      <div class="metric-grid">
        <div><span>Model</span><strong>{record["model_probability"]}</strong></div>
        <div><span>Fair</span><strong>{record["model_fair_odds"]}</strong></div>
        <div><span>Market</span><strong>{record["market_probability"]}</strong></div>
        <div><span>Edge</span><strong>{record["edge"]}</strong></div>
        <div><span>ROI</span><strong>{record["roi"]}</strong></div>
        <div><span>Books</span><strong>{record["books"]}</strong></div>
      </div>

      <div class="support-row">
        <span>Role <b>{record["role"]}</b></span>
        <span>xTD5 <b>{record["xtd5"]}</b></span>
      </div>
      {flags_html}
    </article>
    """


def render_section(title, subtitle, records):
    section_records = [r for r in records if r["tier"] == title]
    section_records.sort(key=lambda r: (r["edge_num"], r["roi_num"]), reverse=True)

    if section_records:
        body = "\n".join(render_card(r) for r in section_records)
    else:
        body = '<div class="empty">No players in this section.</div>'

    slug = title.lower().replace(" ", "-")
    return f"""
    <section id="{slug}" class="section">
      <div class="section-head">
        <div>
          <h2>{title}</h2>
          <p>{subtitle}</p>
        </div>
        <span class="count">{len(section_records)}</span>
      </div>
      <div class="cards">
        {body}
      </div>
    </section>
    """


def build_dashboard(board, season, week, updated_at):
    records = [row_to_record(row) for _, row in board.iterrows()]

    counts = {
        tier: sum(r["tier"] == tier for r in records)
        for tier in ["STRONG VALUE", "ACTIONABLE", "WATCHLIST", "REVIEW ONLY"]
    }

    sections = "\n".join([
        render_section("STRONG VALUE", "7+ pp edge · 3+ books", records),
        render_section("ACTIONABLE", "4–7 pp edge · 3+ books", records),
        render_section("WATCHLIST", "2–4 pp edge · 2+ books", records),
        render_section("REVIEW ONLY", "Positive edge, but thin or lower-confidence market", records),
    ])

    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<meta name="theme-color" content="#070b14">
<title>NFL Anytime TD · Week {week}</title>
<style>
:root {{
  --bg:#070b14;
  --panel:#0e1526;
  --panel2:#121b30;
  --line:#23304d;
  --text:#f4f7fb;
  --muted:#91a0b8;
  --green:#3ddc97;
  --accent:#73a7ff;
  --orange:#f08b5d;
}}
* {{ box-sizing:border-box; }}
html {{ scroll-behavior:smooth; }}
body {{
  margin:0;
  background:var(--bg);
  color:var(--text);
  font-family:Inter,system-ui,-apple-system,Segoe UI,sans-serif;
}}
.wrap {{
  max-width:1180px;
  margin:0 auto;
  padding:22px 16px 64px;
}}
.hero {{
  padding:22px;
  border:1px solid var(--line);
  border-radius:20px;
  background:linear-gradient(145deg,#111b31,#0b1020);
}}
.eyebrow {{
  color:var(--accent);
  font-weight:800;
  font-size:.76rem;
  letter-spacing:.12em;
}}
h1 {{
  margin:7px 0 4px;
  font-size:clamp(1.8rem,5vw,3rem);
}}
.subtitle {{
  color:var(--muted);
  margin:0;
}}
.summary {{
  display:grid;
  grid-template-columns:repeat(4,1fr);
  gap:10px;
  margin-top:18px;
}}
.summary a {{
  color:inherit;
  text-decoration:none;
  padding:13px 14px;
  border:1px solid var(--line);
  background:rgba(255,255,255,.025);
  border-radius:13px;
}}
.summary span {{
  display:block;
  color:var(--muted);
  font-size:.74rem;
}}
.summary strong {{
  font-size:1.35rem;
}}
.updated {{
  color:var(--muted);
  font-size:.78rem;
  margin-top:14px;
}}
.section {{
  margin-top:28px;
}}
.section-head {{
  display:flex;
  justify-content:space-between;
  align-items:end;
  gap:12px;
  margin-bottom:12px;
}}
.section h2 {{
  margin:0;
  font-size:1.2rem;
}}
.section p {{
  color:var(--muted);
  margin:4px 0 0;
  font-size:.86rem;
}}
.count {{
  min-width:38px;
  height:38px;
  display:grid;
  place-items:center;
  border-radius:50%;
  background:var(--panel2);
  border:1px solid var(--line);
  font-weight:800;
}}
.cards {{
  display:grid;
  grid-template-columns:repeat(2,minmax(0,1fr));
  gap:12px;
}}
.player-card {{
  background:var(--panel);
  border:1px solid var(--line);
  border-radius:16px;
  padding:16px;
}}
.card-top {{
  display:flex;
  justify-content:space-between;
  gap:16px;
}}
.player {{
  font-size:1.1rem;
  font-weight:850;
}}
.meta,.book {{
  color:var(--muted);
  font-size:.78rem;
  margin-top:3px;
}}
.best-price {{
  text-align:right;
}}
.price {{
  font-size:1.28rem;
  font-weight:900;
  color:var(--green);
}}
.metric-grid {{
  margin-top:15px;
  display:grid;
  grid-template-columns:repeat(6,1fr);
  gap:8px;
  border-top:1px solid var(--line);
  padding-top:13px;
}}
.metric-grid span {{
  display:block;
  color:var(--muted);
  font-size:.67rem;
}}
.metric-grid strong {{
  display:block;
  margin-top:2px;
  font-size:.9rem;
}}
.support-row {{
  display:flex;
  gap:18px;
  margin-top:13px;
  color:var(--muted);
  font-size:.78rem;
}}
.support-row b {{
  color:var(--text);
}}
.tag {{
  margin-left:7px;
  padding:2px 6px;
  border-radius:6px;
  background:#17243d;
  color:#b9d0ff;
  font-size:.65rem;
}}
.flags {{
  color:var(--orange);
  font-size:.72rem;
  margin-top:10px;
}}
.empty {{
  color:var(--muted);
  border:1px dashed var(--line);
  border-radius:14px;
  padding:18px;
  text-align:center;
}}
.note {{
  margin-top:32px;
  padding:14px 16px;
  border:1px solid var(--line);
  border-radius:14px;
  color:var(--muted);
  font-size:.78rem;
  line-height:1.5;
}}
@media (max-width:800px) {{
  .summary {{ grid-template-columns:repeat(2,1fr); }}
  .cards {{ grid-template-columns:1fr; }}
  .metric-grid {{
    grid-template-columns:repeat(3,1fr);
    row-gap:13px;
  }}
}}
</style>
</head>
<body>
<main class="wrap">
  <header class="hero">
    <div class="eyebrow">NFL ANYTIME TOUCHDOWN</div>
    <h1>Week {week} Value Board</h1>
    <p class="subtitle">{season} · Best available ATD prices vs model probabilities</p>

    <div class="summary">
      <a href="#strong-value"><span>Strong Value</span><strong>{counts["STRONG VALUE"]}</strong></a>
      <a href="#actionable"><span>Actionable</span><strong>{counts["ACTIONABLE"]}</strong></a>
      <a href="#watchlist"><span>Watchlist</span><strong>{counts["WATCHLIST"]}</strong></a>
      <a href="#review-only"><span>Review Only</span><strong>{counts["REVIEW ONLY"]}</strong></a>
    </div>

    <div class="updated">Last generated: {html.escape(updated_at)}</div>
  </header>

  {sections}

  <div class="note">
    <strong>How to read this:</strong>
    Model is the estimated probability of an anytime TD. Fair is the model-implied American price.
    Edge is model probability minus sportsbook implied probability. ROI is model-estimated expected
    return at the displayed price. Review-only candidates typically lack market depth or supporting context.
  </div>
</main>
</body>
</html>
"""


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--season", type=int, required=True)
    parser.add_argument("--week", type=int, required=True)
    parser.add_argument("--output-dir", default="docs")
    args = parser.parse_args()

    board_path = (
        ROOT / "data" / "output" /
        f"{args.season}_week_{args.week}_atd_value_board.csv"
    )

    if not board_path.exists():
        raise FileNotFoundError(
            f"{board_path}\nRun build_atd_value_board.py first."
        )

    board = pd.read_csv(board_path)

    out_dir = ROOT / args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    updated_at = datetime.now().astimezone().strftime("%Y-%m-%d %I:%M %p %Z")
    page = build_dashboard(board, args.season, args.week, updated_at)

    (out_dir / "index.html").write_text(page, encoding="utf-8")
    (out_dir / ".nojekyll").write_text("", encoding="utf-8")

    records = [row_to_record(row) for _, row in board.iterrows()]
    payload = {
        "season": args.season,
        "week": args.week,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "players": records,
    }
    (out_dir / "dashboard.json").write_text(
        json.dumps(payload, indent=2),
        encoding="utf-8",
    )

    print("=" * 72)
    print(" WEB DASHBOARD BUILT")
    print("=" * 72)
    print(f" Season/Week : {args.season} / {args.week}")
    print(f" Players     : {len(board)}")
    print(f" HTML        : {out_dir / 'index.html'}")
    print(f" JSON        : {out_dir / 'dashboard.json'}")
    print("=" * 72)


if __name__ == "__main__":
    main()
