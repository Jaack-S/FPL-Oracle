import sys
from pathlib import Path

# Ensures FPL-Oracle project root is always in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
SEASONS = [
    # pre-2019 contains too many data quirks e.g. no xG etc. We may want to include it one day
    # if we can get understat working
    # 2019-20 contains 47 gameweeks, covid happened, weird namings. Drop it.
    # "2016-17",
    # "2017-18",
    # "2018-19",
    # "2019-20",

    # no understat xG/xA data on a per-player level
    # "2020-21",

    # Rolling cross-validation seasons for train/val
    "2021-22",
    "2022-23",
    "2023-24",
    # Test/holdout
    "2024-25",
    "2025-26"
]

DATA_DIR = Path("data")
MODELS_DIR = Path("models")
RAW_OUTPUT_DIR = DATA_DIR / "raw/vaastav"

POSITION_MAP = {1: "GK", 2: "DEF", 3: "MID", 4: "FWD"}

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