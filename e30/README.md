# Fantasy Football Analytics — AIM111 Exercise 30

An interactive NFL fantasy football dashboard created with **Python, Streamlit, and pandas** for Exercise 30. Analyze regular-season QB/RB/WR/TE performance using public nflverse weekly player statistics.

## Features
- Choose NFL season, position(s), team, and week range
- Switch among PPR points, standard points, targets, carries, receptions, and yardage metrics
- Interactive player leaderboard, bar chart, and sortable table
- Compare up to eight players on a weekly line chart
- Explore recent PPR momentum (last three recorded weeks versus up to three preceding weeks)
- Upload a compatible CSV as an alternative to public data download
- Download filtered results as CSV

## Install and run (Windows)

```bat
py -m venv .venv
.venv\Scripts\python -m pip install -r requirements.txt
.venv\Scripts\python -m streamlit run fantasyanalytics.py
```

Open the local address printed by Streamlit. Do not commit `.venv`.

## Data

By default, the app fetches the selected season from the [nflverse `stats_player` release](https://github.com/nflverse/nflverse-data/releases/tag/stats_player) at:

`https://github.com/nflverse/nflverse-data/releases/download/stats_player/stats_player_week_{season}.csv`

Regular season only; offensive positions QB, RB, WR, TE. External data and network access are required unless you upload a local CSV. Files are cached for six hours; nflverse may revise previously published numbers. Dataset availability depends on nflverse.

**Momentum caveat:** Change in recent PPR points is a descriptive historical comparison, not a projected score or breakout guarantee. Missing game weeks are not assigned zero scores. Filters can change the comparison window.

Source: [nflverse](https://github.com/nflverse/nflverse-data). Data credit: nflverse contributors (CC BY 4.0 where applicable; verify source licensing for reuse).

## Deploy on Streamlit Community Cloud

1. Push these files to `dbrokeshaiSCC/AIM111` on branch `main`, under `e30/`.
2. Visit https://share.streamlit.io and choose **Create app** → deploy from GitHub.
3. Set repository to `dbrokeshaiSCC/AIM111`, branch to `main`, entry point `e30/fantasyanalytics.py`.
4. Deploy, then open the resulting public `https://....streamlit.app/` link and test the widgets and download button.

## Assignment submission

**Title:** Fantasy Football Analytics: Identifying Breakout Players and Hidden Trends

**Overview:** This interactive Streamlit dashboard helps fantasy football players explore NFL weekly player statistics, compare performance, and identify recent scoring trends. Users can filter by season, position, team, and week, then view charts, statistics tables, and downloadable data.

**Repository:** https://github.com/dbrokeshaiSCC/AIM111/tree/main/e30

**Live URL:** Add your confirmed public Streamlit URL after deployment.
