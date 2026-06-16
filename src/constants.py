from pathlib import Path

SEASONS = [
    "2016-17",
    "2017-18",
    "2018-19",
    "2019-20",
    "2020-21",
    "2021-22",
    "2022-23",
    "2023-24",
    "2024-25",
    # "2025-26"
]

DATA_DIR = Path("data")
RAW_OUTPUT_DIR = DATA_DIR / "raw/vaastav"

COLUMNS = [
    "season",
    "GW",
    "name",
    "position",
    "team",
    "opponent_team",
    "fixture",
    "round",
    "was_home",
    "value",
    # target columns
    "total_points",
    "points_next_three",
    "points_next_five",
    # game stats
    "minutes",
    "goals_scored",
    "assists",
    "goals_conceded",
    "clean_sheets",
    "bonus",
    "bps",
    "team_a_score",
    "team_h_score",
    "expected_goals",
    "expected_assists",
    "expected_goal_involvements",
    "expected_goals_conceded",
    # secondary game stats
    "own_goals",
    "penalties_missed",
    "penalties_saved",
    "saves",
    "starts",
    "yellow_cards",
    "red_cards",
    # FPL player selection cols
    "selected",
    "transfers_balance",
    "transfers_in",
    "transfers_out",
]

OTHER_COLS = [
    "xP", "creativity", "element", "ict_index", "influence", "kickoff_time",
    "selected", "starts", "threat"
]