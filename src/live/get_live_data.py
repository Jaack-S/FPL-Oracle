import datetime
import sys
import time
from pathlib import Path

import pandas as pd
import requests

# Put the repo root on sys.path so we can resolve `from src import ...` properly
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.common.constants import DATA_DIR, POSITION_MAP
from src.common.features import Features

# Base URL for static FPL data (players, teams, gameweeks)
FPL_BASE_URL = "https://fantasy.premierleague.com/api"
FPL_BOOTSTRAP_URL = f"{FPL_BASE_URL}/bootstrap-static/"
ELEMENT_SUMMARY_URL = f"{FPL_BASE_URL}/element-summary/{{element_id}}/"
HEADERS = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}


def get_data():

    response = requests.get(FPL_BOOTSTRAP_URL, headers=HEADERS)
    response.raise_for_status()

    return response.json()


def current_season() -> str:
    """FPL seasons run August-May; label as e.g. '2025-26'."""
    today = datetime.date.today()
    start_year = today.year if today.month >= 7 else today.year - 1
    return f"{start_year}-{str(start_year + 1)[-2:]}"


def get_player_history(element_id: int) -> list[dict]:
    """Fetch gameweek history for a specific player ID."""
    url = ELEMENT_SUMMARY_URL.format(element_id=element_id)
    response = requests.get(url, headers=HEADERS)
    if response.status_code == 200:
        return response.json().get("history", [])
    return []


# Iterates over player history to produce gameweek-level rows expected by features.py
def extract_player_data(bootstrap_data: dict) -> pd.DataFrame:
    players = bootstrap_data.get("elements", [])

    # Map dictionaries directly instead of merging DataFrames
    teams_map = {t["id"]: t["short_name"] for t in bootstrap_data.get("teams", [])}

    all_history_rows = []
    season = current_season()

    time_start = datetime.datetime.now()
    # Loop over each player ID to pull gameweek history
    for idx, player in enumerate(players, start=1):
        p_id = player["id"]
        web_name = player.get("web_name", "")
        team_id = player.get("team")
        team_name = teams_map.get(team_id, "")
        position = POSITION_MAP.get(player.get("element_type"), "")
        now_cost = player.get("now_cost", 0)
        name = "_".join(
            f"{player.get('first_name', '')} {player.get('second_name', '')}".split()
        )

        print(f"Getting player history for {p_id}...")
        history = get_player_history(p_id)

        for gw_event in history:
            gw_row = gw_event.copy()
            # Metadata required downstream by features.py
            gw_row["element"] = p_id
            gw_row["web_name"] = web_name
            gw_row["team_name"] = team_name
            gw_row["position"] = position
            gw_row["now_cost"] = now_cost
            # Columns matching data/merged_data.csv's schema, so Features.transform()
            # runs the same code path as training
            gw_row["name"] = name
            gw_row["season"] = season
            gw_row["team"] = team_id
            gw_row["GW"] = gw_row.get("round")
            all_history_rows.append(gw_row)

        # TODO: this seems quite slow tbh.
        time.sleep(0.0001)  # Rate limiting delay

    time_end = datetime.datetime.now()
    duration_s = (time_end - time_start).seconds
    print(f"Fetching player data took {duration_s} seconds")

    df = pd.DataFrame(all_history_rows)

    # The FPL API returns these as strings; Features.transform() needs them numeric
    for col in ["expected_goals", "expected_assists", "expected_goals_conceded"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    return df


if __name__ == "__main__":
    if not (DATA_DIR / "live").exists():
        (DATA_DIR / "live").mkdir(parents=True, exist_ok=True)

    raw_data = get_data()
    print("Got raw data")
    print("Getting player data...")
    df_players = extract_player_data(raw_data)

    # Save live inference data separately from training data
    # TODO: We should save this with a timestamp so that we can track things over time.
    df_players.to_csv(DATA_DIR / "live/live_fpl_data.csv", index=False)
    print(
        f"Fetched {len(df_players)} total gameweek records for {len(raw_data['elements'])} players and saved to live_fpl_data.csv"
    )

    print("Building features...")
    df_features = Features().transform(df_players)
    df_features.to_csv(DATA_DIR / "live/data_with_features.csv", index=False)
    print(f"Feature data saved to {DATA_DIR / 'live/data_with_features.csv'}")
