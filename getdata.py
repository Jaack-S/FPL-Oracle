import pandas as pd

from constants import RAW_OUTPUT_DIR, SEASONS

BASE_URL = "https://raw.githubusercontent.com/vaastav/Fantasy-Premier-League/master/data"


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

        df.to_csv(out_path, index=False)
        print(f"{len(df)} rows saved to {out_path}")


if __name__ == "__main__":
    download_vaastav()








import numpy as np
import pandas as pd


df = pd.read_csv('data/merged_data.csv')

df = df.sort_values(by=['name', 'GW']).reset_index(drop=True)

# Calculate total points in the next 3 gameweeks (GW1 + GW2 + GW3)
df['target_points_next_3_gws'] = (
    df.groupby('name')['total_points'].shift(-1) +
    df.groupby('name')['total_points'].shift(-2) +
    df.groupby('name')['total_points'].shift(-3)
)

# Calculate total points in the next 5 gameweeks
df['target_points_next_5_gws'] = (
    df.groupby('name')['total_points'].shift(-1) +
    df.groupby('name')['total_points'].shift(-2) +
    df.groupby('name')['total_points'].shift(-3) +
    df.groupby('name')['total_points'].shift(-4) +
    df.groupby('name')['total_points'].shift(-5)
)