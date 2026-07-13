import pandas as pd

from clean import add_target_columns, clean
from constants import DATA_DIR, RAW_OUTPUT_DIR, SEASONS, UNDERSTAT_SEASONS

if __name__ == "__main__":
    data = [] 

    for season in SEASONS:
        path = RAW_OUTPUT_DIR / f"{season}_merged_gw.csv"
        if path.exists():
            df = pd.read_csv(path)
            df["season"] = season
            data.append(df)
        else:
            raise FileNotFoundError(f"File not found: {path}")
        
    if not data:
        raise ValueError("No data found to merge. Please ensure that the CSV files exist in the specified directory.")
        
    merged_df = pd.concat(data, ignore_index=True)
    print(f"Merged {len(data)} seasons of data with a total of {len(merged_df)} rows.")
    print(f"There are {merged_df['name'].nunique()} unique players in the merged data.")
    print(f"There are {merged_df['season'].nunique()} unique seasons in the merged data.")
    
    # Sort the data to make sure the rolling(n).sum() is correct, and reset the index for neatness
    merged_df.sort_values(by=["season", "name", "GW"], ascending=[True, True, True], inplace=True)
    merged_df.reset_index(drop=True, inplace=True)

    # Clean the data
    merged_df = clean(merged_df)

    # Now create target columns
    merged_df = add_target_columns(merged_df)
    print("Added target columns: points_next_three and points_next_five.")

    # Merge in Understat xG/xA features (run src/understat.py first to produce this file)
    understat_path = DATA_DIR / "understat_data.csv"
    if understat_path.exists():
        understat_df = pd.read_csv(understat_path)
        merged_df = merged_df.merge(understat_df, on=["season", "element", "GW"], how="left")

        # A null here means one of two different things: no attacking involvement that
        # gameweek (true zero, only possible for seasons Understat actually covers), or
        # no Understat coverage for the season at all (genuinely missing, not zero).
        understat_cols = [c for c in understat_df.columns if c.startswith("understat_")]
        covered = merged_df["season"].isin(UNDERSTAT_SEASONS)
        merged_df.loc[covered, understat_cols] = merged_df.loc[covered, understat_cols].fillna(0)
        print(f"Merged Understat features: {understat_cols}")
    else:
        print(f"No Understat data found at {understat_path} - run src/understat.py first to include it.")

    # Save the merged data to a new CSV file
    merged_df.to_csv(DATA_DIR / "merged_data.csv", index=False)
    print(f"Merged data saved to {DATA_DIR / 'merged_data.csv'}")
