from pathlib import Path

import pandas as pd

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
BASE_URL = "https://raw.githubusercontent.com/vaastav/Fantasy-Premier-League/master/data"
OUTPUT_DIR = Path("data/raw/vaastav")


def download_vaastav():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    for season in SEASONS:
        url = f"{BASE_URL}/{season}/gws/merged_gw.csv"
        out_path = OUTPUT_DIR / f"{season}_merged_gw.csv"

        print(f"Downloading {season}...")
        try:
            df = pd.read_csv(url)
        except UnicodeDecodeError:
            try:
                df = pd.read_csv(url, encoding="latin1")
            except Exception as e:
                print(f"Error downloading {season}: {e}")
                continue

        df.to_csv(out_path, index=False)
        print(f"{len(df)} rows saved to {out_path}")


if __name__ == "__main__":
    download_vaastav()
