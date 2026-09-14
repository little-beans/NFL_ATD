# NFL ATD Web Dashboard Patch V2

This patch turns the existing ATD value-board CSV into a clean mobile-friendly
website and optionally posts the permanent dashboard link into Discord.

## Adds

- `scripts/build_web_dashboard.py`
- `scripts/post_discord_dashboard.py`
- `.github/workflows/pages.yml`
- `.env.example`
- `tests/test_web_dashboard.py`

## 1. Build locally

After your normal model + odds + value-board run:

```powershell
python scripts/build_web_dashboard.py --season 2026 --week 2
```

Generated files:

```text
docs\index.html
docs\dashboard.json
docs\.nojekyll
```

Open `docs\index.html` in a browser to preview it.

## 2. Enable GitHub Pages once

In the `little-beans/NFL_ATD` GitHub repository:

1. Open **Settings**
2. Open **Pages**
3. Under **Build and deployment**, select **GitHub Actions**

## 3. Commit and deploy

```powershell
git add docs scripts/build_web_dashboard.py scripts/post_discord_dashboard.py .github/workflows/pages.yml .env.example tests/test_web_dashboard.py
git commit -m "Add ATD web dashboard"
git push
```

Expected project URL:

```text
https://little-beans.github.io/NFL_ATD/
```

Use the exact URL shown by GitHub Pages after deployment.

## 4. Weekly refresh

```powershell
python scripts/score_week_unified.py --season 2026 --week 2
python scripts/fetch_atd_odds.py --season 2026 --week 2 --refresh
python scripts/build_atd_value_board.py --season 2026 --week 2
python scripts/build_web_dashboard.py --season 2026 --week 2

git add docs
git commit -m "Refresh Week 2 ATD dashboard"
git push
```

The same URL updates after the Pages deployment completes.

## 5. Optional Discord link

Put these only in your local `.env`:

```text
DISCORD_ATD_WEBHOOK_URL=your_real_webhook_url
ATD_DASHBOARD_URL=https://little-beans.github.io/NFL_ATD/
```

Then run:

```powershell
python scripts/post_discord_dashboard.py --season 2026 --week 2
```

Discord receives a short Strong / Actionable / Watchlist summary plus a link
to the dashboard.

Never commit `.env`.
