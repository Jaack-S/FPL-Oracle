import sys
from pathlib import Path

import pandas as pd
import requests

# Put the repo root on sys.path so we can resolve `from src import ...` properly
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.common.constants import DATA_DIR

# Base URL for static FPL data (players, teams, gameweeks)
FPL_BOOTSTRAP_URL = "https://fantasy.premierleague.com/api/bootstrap-static/"

def get_data():
    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}

    response = requests.get(FPL_BOOTSTRAP_URL, headers=headers)
    response.raise_for_status()

    return response.json()

def extract_player_data(data):
    # "elements" contains all player stats
    players_df = pd.DataFrame(data["elements"])

    # "element_types" contains the mappings 1 = GKP, 2 = DEF, 3 = MID, 4 = FWD
    positions_df = pd.DataFrame(data["element_types"])[["id", "singular_name_short"]]
    positions_df.rename(
        columns={"id": "element_type", "singular_name_short": "position"},
        inplace=True,)
    
    # "teams" contains team name mappings
    teams_df = pd.DataFrame(data["teams"])[["id", "short_name"]]
    teams_df.rename(
        columns={"id": "team", "short_name": "team_name"}, inplace=True
    )

    # Merge position and team names into the main players DataFrame
    players_df = players_df.merge(positions_df, on="element_type", how="left")
    players_df = players_df.merge(teams_df, on="team", how="left")

    return players_df

if __name__ == "__main__":
    raw_data = get_data()
    df_players = extract_player_data(raw_data)

    # Save live inference data separately from training data
    # TODO: We should save this with a timestamp so that we can track things over time.
    df_players.to_csv(DATA_DIR / "live/live_fpl_data.csv", index=False)
    print(f"Fetched {len(df_players)} players and saved to live_fpl_data.csv")
