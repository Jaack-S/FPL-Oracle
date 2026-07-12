import pandas as pd

from src.constants import RAW_OUTPUT_DIR, SEASONS

BASE_URL = "https://raw.githubusercontent.com/vaastav/Fantasy-Premier-League/master/data"

POSITION_MAP = {1: "GK", 2: "DEF", 3: "MID", 4: "FWD"}


def backfill_position_and_team(df: pd.DataFrame, season: str) -> pd.DataFrame:
    """
    2016-17 through 2018-19 merged_gw.csv files have no `position` or `team`
    columns. Reconstruct them from that season's players_raw.csv (element_type,
    team id) and the repo's master_team_list.csv (season, team id -> team name).
    """
    players_url = f"{BASE_URL}/{season}/players_raw.csv"
    try:
        players = pd.read_csv(players_url)
    except UnicodeDecodeError:
        players = pd.read_csv(players_url, encoding="latin1")

    master_teams = pd.read_csv(f"{BASE_URL}/master_team_list.csv")
    season_teams = master_teams[master_teams["season"] == season].set_index("team")["team_name"]

    players = players.assign(
        position=players["element_type"].map(POSITION_MAP),
        team=players["team"].map(season_teams),
    )

    return df.merge(
        players[["id", "position", "team"]].rename(columns={"id": "element"}),
        on="element",
        how="left",
    )


def download_vaastav():
    RAW_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    for season in SEASONS:
        url = f"{BASE_URL}/{season}/gws/merged_gw.csv"
        out_path = RAW_OUTPUT_DIR / f"{season}_merged_gw.csv"

        print(f"Downloading {season}...")
        try:
            df = pd.read_csv(url)
        except UnicodeDecodeError:
            try:
                df = pd.read_csv(url, encoding="latin1")
            except Exception as e:
                print(f"Error downloading {season}: {e}")
                continue

        if "position" not in df.columns or "team" not in df.columns:
            print(f"  Backfilling position/team for {season}...")
            df = backfill_position_and_team(df, season)

        df.to_csv(out_path, index=False)
        print(f"{len(df)} rows saved to {out_path}")


if __name__ == "__main__":
    download_vaastav()
