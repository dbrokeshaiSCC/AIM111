"""AIM111 Exercise 30 — NFL fantasy football analytics using public nflverse data."""
from __future__ import annotations

from datetime import date

import pandas as pd
import streamlit as st

DATA_URL = (
    "https://github.com/nflverse/nflverse-data/releases/download/"
    "stats_player/stats_player_week_{season}.csv"
)
METRICS = {
    "PPR fantasy points": "fantasy_points_ppr",
    "Standard fantasy points": "fantasy_points",
    "Targets": "targets",
    "Carries": "carries",
    "Receptions": "receptions",
    "Receiving yards": "receiving_yards",
    "Rushing yards": "rushing_yards",
    "Passing yards": "passing_yards",
}
POSITIONS = ["QB", "RB", "WR", "TE"]


@st.cache_data(ttl=6 * 60 * 60, show_spinner="Downloading NFL statistics from nflverse…")
def fetch_season(season: int) -> pd.DataFrame:
    """Fetch a single season's weekly player CSV; cache for six hours."""
    return pd.read_csv(DATA_URL.format(season=season), low_memory=False)


def prepare_data(raw: pd.DataFrame) -> pd.DataFrame:
    """Normalize column variants, validate schema, and filter to regular season."""
    data = raw.copy()
    aliases = {
        "player_display_name": ["player_display_name", "player_name", "name"],
        "team": ["team", "recent_team"],
    }
    for canonical, possibilities in aliases.items():
        if canonical not in data.columns:
            match = next((col for col in possibilities if col in data.columns), None)
            if match:
                data[canonical] = data[match]
    required = {"player_display_name", "team", "position", "week", "fantasy_points_ppr"}
    missing = required - set(data.columns)
    if missing:
        raise ValueError("Dataset is missing required fields: " + ", ".join(sorted(missing)))
    if "season_type" in data.columns:
        season_types = data["season_type"].map(lambda value: str(value).strip().upper())
        data = data.loc[season_types == "REG"].copy()
    data["position"] = data["position"].astype("string").str.upper().str.strip()
    data = data.loc[data["position"].isin(POSITIONS)].copy()
    data["week"] = pd.to_numeric(data["week"], errors="coerce")
    # Missing optional metrics should be shown as unavailable, not fabricated zeroes.
    for col in METRICS.values():
        if col not in data.columns:
            data[col] = float("nan")
        data[col] = pd.to_numeric(data[col], errors="coerce")
    data = data.dropna(subset=["player_display_name", "week"])
    data["player_display_name"] = data["player_display_name"].astype(str)
    data["team"] = data["team"].fillna("Unknown").astype(str)
    data["week"] = data["week"].astype(int)
    data = data.loc[data["week"] >= 1].copy()
    # Keep player-week rows distinct if the source has duplicates.
    if "player_id" in data.columns:
        keys = ["player_id", "week"]
    else:
        keys = ["player_display_name", "week"]
    data = data.drop_duplicates(subset=keys, keep="last")
    return data


def aggregate_players(data: pd.DataFrame, metric: str) -> pd.DataFrame:
    """Build leaderboards based on the currently filtered player-week rows."""
    result = (
        data.groupby(["player_display_name", "position"], as_index=False)
        .agg(
            team=("team", "last"),
            games=("week", "nunique"),
            total=(metric, "sum"),
            avg=(metric, "mean"),
            targets=("targets", "sum"),
            carries=("carries", "sum"),
        )
    )
    return result.sort_values("total", ascending=False)


def breakout_table(data: pd.DataFrame) -> pd.DataFrame:
    """Compare last 3 available weeks to preceding 3 for players with 4+ games.

    A descriptive momentum indicator, NOT a prediction or adjusted projection.
    """
    rows = []
    for (player, position), group in data.groupby(["player_display_name", "position"]):
        weekly = pd.DataFrame(
            group.groupby("week", as_index=False)["fantasy_points_ppr"].sum()
        )
        weekly = weekly.sort_values("week")
        if len(weekly) < 4:
            continue
        recent = weekly.tail(3)["fantasy_points_ppr"].mean()
        prior = weekly.iloc[:-3].tail(3)["fantasy_points_ppr"].mean()
        if pd.isna(prior):
            continue
        rows.append({
            "Player": player, "Position": position,
            "Recent avg (PPR)": round(recent, 2),
            "Prior avg (PPR)": round(prior, 2),
            "Change (PPR/week)": round(recent - prior, 2),
            "Games": len(weekly),
        })
    if not rows:
        return pd.DataFrame(columns=["Player", "Position", "Recent avg (PPR)",
                                     "Prior avg (PPR)", "Change (PPR/week)", "Games"])
    return pd.DataFrame(rows).sort_values("Change (PPR/week)", ascending=False)


def main() -> None:
    st.set_page_config(page_title="Fantasy Football Analytics", page_icon="🏈", layout="wide")
    st.title("🏈 Fantasy Football Analytics")
    st.caption("Explore NFL player performance, compare fantasy production, and spot recent momentum.")

    st.sidebar.header("Filters")
    current = date.today().year
    seasons = list(range(current, 2021, -1))
    season = st.sidebar.selectbox("NFL season", seasons, index=0)
    uploaded = st.sidebar.file_uploader("Or upload an nflverse-style CSV", type=["csv"])
    st.sidebar.caption("Public data: nflverse player statistics (regular season).")

    try:
        if uploaded is not None:
            raw = pd.read_csv(uploaded, low_memory=False)
            source = "Uploaded CSV"
        else:
            raw = fetch_season(season)
            source = f"nflverse {season}"
        data = prepare_data(raw)
    except Exception as exc:
        st.error(f"Unable to load player statistics: {exc}")
        st.info("Check your network connection or upload a compatible CSV using the sidebar.")
        st.stop()

    if data.empty:
        st.warning("No regular-season QB/RB/WR/TE statistics were found in the selected data.")
        st.stop()

    position = st.sidebar.multiselect("Position", POSITIONS, default=POSITIONS)
    if not position:
        st.warning("Select at least one position in the sidebar.")
        st.stop()
    data = data[data["position"].isin(position)].copy()
    if data.empty:
        st.warning("No player records match the selected positions.")
        st.stop()
    teams = sorted(data["team"].unique().tolist())
    team = st.sidebar.selectbox("Team", ["All teams"] + teams)
    if team != "All teams":
        data = data.loc[data["team"] == team].copy()
    if data.empty:
        st.warning("No player records match the selected team and positions.")
        st.stop()
    min_week, max_week = int(data["week"].min()), int(data["week"].max())
    selected_weeks = st.sidebar.slider("Week range", min_week, max_week,
                                       (min_week, max_week)) if min_week < max_week else (min_week, max_week)
    data = data.loc[data["week"].between(*selected_weeks)].copy()
    if data.empty:
        st.warning("No records match your filters.")
        st.stop()
    metric_label = st.sidebar.selectbox("Comparison metric", list(METRICS))
    metric = METRICS[metric_label]
    if data[metric].notna().sum() == 0:
        st.warning(f"{metric_label} is unavailable in the selected dataset.")
        st.stop()
    top_n = st.sidebar.slider("Leaderboard size", 5, 30, 10, step=5)

    leaderboard = aggregate_players(data, metric)
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Players", len(leaderboard))
    m2.metric("Teams", data["team"].nunique())
    m3.metric("Weeks in view", data["week"].nunique())
    m4.metric("Fantasy PPR points", f"{data['fantasy_points_ppr'].sum():,.1f}")

    tab1, tab2, tab3 = st.tabs(["📊 Leaderboard", "📈 Player comparison", "🚀 Momentum explorer"])
    with tab1:
        st.subheader(f"Top {top_n} players — {metric_label}")
        best = leaderboard.head(top_n)
        st.bar_chart(best.set_index("player_display_name")["total"], horizontal=True)
        st.dataframe(
            leaderboard.rename(columns={
                "player_display_name": "Player", "position": "Position", "team": "Team",
                "games": "Games", "total": f"Total {metric_label}",
                "avg": f"Per-game {metric_label}", "targets": "Targets", "carries": "Carries"
            }),
            hide_index=True, use_container_width=True,
        )
    with tab2:
        st.subheader("Weekly player comparison")
        available = leaderboard["player_display_name"].tolist()
        defaults = available[:min(3, len(available))]
        selected = st.multiselect("Choose up to 8 players", available, default=defaults, max_selections=8)
        if selected:
            weekly = (
                data[data["player_display_name"].isin(selected)]
                .groupby(["week", "player_display_name"], as_index=False)[metric].sum()
                .pivot(index="week", columns="player_display_name", values=metric)
                .sort_index()
            )
            st.line_chart(weekly)
            st.caption("Missing weeks are left blank; values are not forward-filled.")
        else:
            st.info("Select players to display their weekly performance.")
    with tab3:
        st.subheader("Recent fantasy point momentum")
        st.write("Compares each player's average PPR points over their last 3 recorded weeks "
                 "with up to 3 immediately preceding recorded weeks in the selected range. "
                 "At least 4 recorded games are required.")
        momentum = breakout_table(data)
        if momentum.empty:
            st.info("Not enough weeks of data for a momentum comparison. Expand the week range.")
        else:
            st.dataframe(momentum, hide_index=True, use_container_width=True)
            st.bar_chart(momentum.head(10).set_index("Player")["Change (PPR/week)"], horizontal=True)
        st.caption("Momentum is descriptive, not a prediction. Missed games are not treated as zero-point weeks.")

    st.divider()
    csv_data = data.to_csv(index=False).encode("utf-8")
    st.download_button("⬇ Download filtered player data (CSV)", data=csv_data,
                       file_name="fantasy_player_stats.csv", mime="text/csv")
    st.caption(f"Data source: {source}. Credit: nflverse (https://github.com/nflverse/nflverse-data). "
               "Statistics may be updated or corrected by the source. For educational use.")


if __name__ == "__main__":
    main()
