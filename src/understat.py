# ABOUTME: Resolves Understat match data to FPL player identity and gameweek, then
# ABOUTME: aggregates xG/xA/etc into per-(season, element, GW) features.
import re
import unicodedata

import pandas as pd

from constants import DATA_DIR, ID_DICT_SEASONS, RAW_OUTPUT_DIR, UNDERSTAT_RAW_DIR, UNDERSTAT_SEASONS

FPL_DATA_URL = "https://raw.githubusercontent.com/vaastav/Fantasy-Premier-League/master/data"

UNDERSTAT_STAT_COLUMNS = [
    "goals", "shots", "xG", "xA", "assists", "key_passes",
    "npg", "npxG", "xGChain", "xGBuildup",
]


def _normalize_name(name: str) -> str:
    name = unicodedata.normalize("NFKD", name).encode("ascii", "ignore").decode()
    return re.sub(r"[^a-zA-Z]", "", name).lower()


def _to_fpl_season(understat_season: int) -> str:
    """Understat's `season` column is the year the season started (e.g. 2021 for 2021-22)."""
    return f"{understat_season}-{str(understat_season + 1)[-2:]}"


def _load_players_raw(season: str) -> pd.DataFrame:
    return pd.read_csv(f"{FPL_DATA_URL}/{season}/players_raw.csv")


def load_all_matches() -> pd.DataFrame:
    """
    Each per-player file downloaded per season folder is that player's entire
    Understat match history, so the same match shows up in every folder a player
    appears in. Concat everything and de-duplicate on (understat_id, match id),
    then use each row's own `season` column to find its true FPL season - the
    season folder it happened to be downloaded from is irrelevant.
    """
    frames = [pd.read_csv(UNDERSTAT_RAW_DIR / f"{season}_matches.csv") for season in UNDERSTAT_SEASONS]
    matches = pd.concat(frames, ignore_index=True).drop_duplicates(subset=["understat_id", "id"])
    matches["season"] = matches["season"].map(_to_fpl_season)
    return matches[matches["season"].isin(UNDERSTAT_SEASONS)]


def build_id_mapping() -> dict[int, int]:
    """
    Understat_ID -> FPL permanent player `code`, built from the seasons that ship an
    id_dict.csv (2021-22, 2022-23). Understat_ID is stable for a player across seasons,
    so this mapping is reused for seasons without their own id_dict.
    """
    mapping: dict[int, int] = {}
    for season in ID_DICT_SEASONS:
        id_dict = pd.read_csv(UNDERSTAT_RAW_DIR / f"{season}_id_dict.csv")
        fpl_id_to_code = _load_players_raw(season).set_index("id")["code"]
        codes = id_dict["FPL_ID"].map(fpl_id_to_code)
        for understat_id, code in zip(id_dict["Understat_ID"], codes):
            if pd.notna(code):
                mapping[int(understat_id)] = int(code)
    return mapping


def resolve_player_codes(matches: pd.DataFrame, season: str, id_mapping: dict[int, int]) -> pd.DataFrame:
    """
    Attach FPL permanent `code` to each Understat row: first via the carried
    Understat_ID mapping, then via normalized exact name matching against that
    season's players_raw for whoever is left. Rows that still can't be resolved
    are dropped (logged, not silently zero-filled).
    """
    matches = matches.copy()

    players_raw = _load_players_raw(season)
    name_to_code = dict(zip(
        (players_raw["first_name"] + players_raw["second_name"]).map(_normalize_name),
        players_raw["code"],
    ))

    by_id = matches["understat_id"].map(id_mapping)
    by_name = matches["understat_name"].map(lambda name: name_to_code.get(_normalize_name(name)))
    matches["code"] = by_id.combine_first(by_name)

    n_total = matches["understat_id"].nunique()
    n_unmatched = matches.loc[matches["code"].isna(), "understat_id"].nunique()
    if n_unmatched:
        print(f"  {season}: {n_unmatched}/{n_total} Understat players could not be matched to an FPL code")

    return matches.dropna(subset=["code"]).astype({"code": int})


def assign_gameweeks(matches: pd.DataFrame, season: str) -> pd.DataFrame:
    """
    Map each Understat match to (element, GW) via that season's raw per-fixture FPL
    data, joining on player code/element and match date.
    """
    players_raw = _load_players_raw(season)
    code_to_element = players_raw.set_index("code")["id"]

    matches = matches.copy()
    matches["element"] = matches["code"].map(code_to_element)
    matches = matches.dropna(subset=["element"]).astype({"element": int})
    matches["date"] = pd.to_datetime(matches["date"]).dt.date

    fixtures = pd.read_csv(
        RAW_OUTPUT_DIR / f"{season}_merged_gw.csv",
        usecols=["element", "kickoff_time", "GW"],
    )
    fixtures["date"] = pd.to_datetime(fixtures["kickoff_time"]).dt.date
    fixtures = fixtures[["element", "date", "GW"]].drop_duplicates()

    merged = matches.merge(fixtures, on=["element", "date"], how="left")
    n_unmatched = merged["GW"].isna().sum()
    if n_unmatched:
        print(f"  {season}: {n_unmatched} match rows could not be matched to a gameweek by date")

    return merged.dropna(subset=["GW"]).astype({"GW": int})


def aggregate_by_gameweek(matches: pd.DataFrame) -> pd.DataFrame:
    """Sum Understat stats within (season, element, GW) to handle double gameweeks."""
    return (
        matches.groupby(["season", "element", "GW"])[UNDERSTAT_STAT_COLUMNS]
        .sum()
        .add_prefix("understat_")
        .reset_index()
    )


def build_understat_features() -> pd.DataFrame:
    id_mapping = build_id_mapping()
    all_matches = load_all_matches()

    season_frames = []
    for season in UNDERSTAT_SEASONS:
        print(f"Resolving Understat data for {season}...")
        matches = all_matches[all_matches["season"] == season].copy()
        matches = resolve_player_codes(matches, season, id_mapping)
        matches = assign_gameweeks(matches, season)
        season_frames.append(matches)

    resolved = pd.concat(season_frames, ignore_index=True)
    return aggregate_by_gameweek(resolved)


if __name__ == "__main__":
    features = build_understat_features()
    out_path = DATA_DIR / "understat_data.csv"
    features.to_csv(out_path, index=False)
    print(f"Understat features saved to {out_path} ({len(features)} rows)")
