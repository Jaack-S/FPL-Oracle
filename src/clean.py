# Import necessary packages
import pandas as pd


def add_target_columns(df: pd.DataFrame) -> pd.DataFrame:
    # group by season and name
    g = df.groupby(["season", "name"])["total_points"]

    # points for nect 3 and 5 gameweeks
    df["points_next_three"] = g.shift(-1) + g.shift(-2) + g.shift(-3)
    df["points_next_five"] = g.shift(-1) + g.shift(-2) + g.shift(-3) + g.shift(-4) + g.shift(-5)

    return df

def clean(df: pd.DataFrame) -> pd.DataFrame:
    # remove 2019-20 season completely
    if 'season' in df.columns:
        df = df[~df['season'].astype(str).str.contains('2019[-_]20')].copy()

    # remove assistant manager data
    if 'position' in df.columns:
        df = df[df['position'] != 'AM'].copy()

    # 2021-22 uses 'GKP' instead of 'GK' for goalkeepers; normalise to one label
    if 'position' in df.columns:
        df['position'] = df['position'].replace('GKP', 'GK')

    # drop assistant manager chip stats - almost entirely null and always 0 when present
    mng_columns = [
        "mng_win", "mng_draw", "mng_loss", "mng_clean_sheets",
        "mng_goals_scored", "mng_underdog_win", "mng_underdog_draw",
    ]
    df = df.drop(columns=[c for c in mng_columns if c in df.columns])

    # make sure names are in form Firstname_Surname
    if 'name' in df.columns:
        df["name"] = df["name"].astype(str).str.strip()
        df["name"] = df["name"].str.replace(r"\s+", "_", regex=True)

    # Deduplicate by GW: if a player appears twice in the same GW, keep the row with most minutes
    df = (
        df.sort_values("minutes", ascending=False)
        .drop_duplicates(subset=["season", "name", "GW"], keep="first")
    )
    
    print('Adding target columns: points in next 3 and 5 gameweeks...')
    # sort data in order
    df = df.sort_values(by=["season", "name", "GW"]).reset_index(drop=True)

    return df
