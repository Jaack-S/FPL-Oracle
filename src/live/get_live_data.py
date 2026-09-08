import sys
from pathlib import Path
import time
import pandas as pd
import requests

# Put the repo root on sys.path so we can resolve `from src import ...` properly
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.common.constants import DATA_DIR

# Base URL for static FPL data (players, teams, gameweeks)
FPL_BASE_URL = "https://fantasy.premierleague.com/api"
FPL_BOOTSTRAP_URL = f"{FPL_BASE_URL}/bootstrap-static/"
ELEMENT_SUMMARY_URL = (
    f"{FPL_BASE_URL}/element-summary/{{element_id}}/"
)
HEADERS = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}
def get_data():

    response = requests.get(FPL_BOOTSTRAP_URL, headers=HEADERS)
    response.raise_for_status()

    return response.json()

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
    teams_map = {
        t["id"]: t["short_name"] for t in bootstrap_data.get("teams", [])
    }
    positions_map = {
        p["id"]: p["singular_name_short"]
        for p in bootstrap_data.get("element_types", [])
    }

    all_history_rows = []

    # Loop over each player ID to pull gameweek history
    for idx, player in enumerate(players, start=1):
        p_id = player["id"]
        web_name = player.get("web_name", "")
        team_name = teams_map.get(player.get("team"), "")
        position = positions_map.get(player.get("element_type"), "")
        now_cost = player.get("now_cost", 0)

        history = get_player_history(p_id)

        for gw_event in history:
            gw_row = gw_event.copy()
            # Metadata required downstream by features.py
            gw_row["element"] = p_id
            gw_row["web_name"] = web_name
            gw_row["team_name"] = team_name
            gw_row["position"] = position
            gw_row["now_cost"] = now_cost
            all_history_rows.append(gw_row)

        time.sleep(0.02)  # Rate limiting delay

    return pd.DataFrame(all_history_rows)

if __name__ == "__main__":
    if not (DATA_DIR / "live").exists():
        (DATA_DIR / "live").mkdir(parents=True, exist_ok=True)

    raw_data = get_data()
    df_players = extract_player_data(raw_data)

    # Save live inference data separately from training data
    # TODO: We should save this with a timestamp so that we can track things over time.
    df_players.to_csv(DATA_DIR / "live/live_fpl_data.csv", index=False)
    print(f"Fetched {len(df_players)} total gameweek records for {len(raw_data['elements'])} players and saved to live_fpl_data.csv")
