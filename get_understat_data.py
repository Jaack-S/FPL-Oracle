# ABOUTME: Downloads Understat per-player match data and the Understat<->FPL id_dict
# ABOUTME: mapping from the vaastav repo, caching both locally for src/understat.py.
import html
import json
import urllib.parse
import urllib.request

import pandas as pd

from src.constants import ID_DICT_SEASONS, UNDERSTAT_RAW_DIR, UNDERSTAT_SEASONS

BASE_URL = "https://raw.githubusercontent.com/vaastav/Fantasy-Premier-League/master/data"
API_URL = "https://api.github.com/repos/vaastav/Fantasy-Premier-League/contents/data"


def list_understat_player_files(season: str) -> list[dict]:
    with urllib.request.urlopen(f"{API_URL}/{season}/understat") as resp:
        entries = json.load(resp)
    # team-level aggregate files (e.g. "understat_Arsenal.csv") carry no player id
    return [e for e in entries if e["name"].endswith(".csv") and not e["name"].startswith("understat_")]


def _download_player_file(entry: dict) -> pd.DataFrame:
    name = entry["name"].removesuffix(".csv")
    player_name, _, understat_id = name.rpartition("_")
    # download_url is percent-encoded for ASCII specials (e.g. "&" -> "%26") but left
    # raw for non-ASCII characters (e.g. "é"); quote with % as safe so we only encode
    # the latter and don't double-encode the former
    url = urllib.parse.quote(entry["download_url"], safe=":/%")
    df = pd.read_csv(url)
    df["understat_id"] = int(understat_id)
    # some filenames contain literal HTML entities (e.g. "Daniel_N&#039;Lundulu")
    df["understat_name"] = html.unescape(player_name.replace("_", " "))
    return df


def download_understat_matches(season: str) -> pd.DataFrame:
    """
    Each per-player file is that player's entire Understat match history, not just
    matches from this season's folder - the file's own `season` column (Understat's
    convention, e.g. 2021 for 2021-22) is what actually distinguishes the season of
    each row, so it must be left untouched here.
    """
    out_path = UNDERSTAT_RAW_DIR / f"{season}_matches.csv"
    if out_path.exists():
        print(f"{season}: using cached {out_path}")
        return pd.read_csv(out_path)

    entries = list_understat_player_files(season)
    print(f"{season}: downloading {len(entries)} player match files...")
    frames = [_download_player_file(e) for e in entries]

    df = pd.concat(frames, ignore_index=True)
    UNDERSTAT_RAW_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    print(f"{season}: saved {len(df)} rows to {out_path}")
    return df


def download_id_dict(season: str) -> pd.DataFrame:
    out_path = UNDERSTAT_RAW_DIR / f"{season}_id_dict.csv"
    if out_path.exists():
        return pd.read_csv(out_path)

    id_dict = pd.read_csv(f"{BASE_URL}/{season}/id_dict.csv")
    id_dict.columns = id_dict.columns.str.strip()
    UNDERSTAT_RAW_DIR.mkdir(parents=True, exist_ok=True)
    id_dict.to_csv(out_path, index=False)
    print(f"{season}: saved id_dict to {out_path}")
    return id_dict


def download_understat():
    for season in UNDERSTAT_SEASONS:
        download_understat_matches(season)
    for season in ID_DICT_SEASONS:
        download_id_dict(season)


if __name__ == "__main__":
    download_understat()
