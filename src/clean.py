# Import necessary packages
import pandas as pd

def clean(df: pd.DataFrame) -> pd.DataFrame:
    # remove 2019-20 season completely
    if 'season' in df.columns:
        df = df[~df['season'].astype(str).str.contains('2019[-_]20')].copy()

    # remove assistant manager data
    if 'position' in df.columns:
        df = df[df['position'] != 'AM'].copy()

    # make sure names are in form Firstname_Surname
    if 'name' in df.columns:
        df["name"] = df["name"].astype(str).str.strip()
        df["name"] = df["name"].str.replace(r"\s+", "_", regex=True)
    
    print('Adding target columns: points in next 3 and 5 gameweeks...')
    # sort data in order
    df = df.sort_values(by=["season", "name", "GW"]).reset_index(drop=True)

    # group by season and name
    g = df.groupby(["season", "name"])["total_points"]

    # points for nect 3 and 5 gameweeks
    df["points_next_three"] = g.shift(-1) + g.shift(-2) + g.shift(-3)
    df["points_next_five"] = g.shift(-1) + g.shift(-2) + g.shift(-3) + g.shift(-4) + g.shift(-5)
    print('Added target columns.')

    return df