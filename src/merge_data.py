import pandas as pd

from constants import DATA_DIR, RAW_OUTPUT_DIR, SEASONS


def rolling_forward_sum(group, window):
    pts = group["total_points"].values
    result = []
    for i in range(len(pts)):
        result.append(pts[i:i+window].sum())
    return pd.Series(result, index=group.index)

def clean_data(df):
    # Make the name clean
    df["name"] = df["name"].str.strip()
    df["name"] = df["name"].str.replace(r"\s+", "_", regex=True) # replace blank space
    return df



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
    merged_df = clean_data(merged_df)

    # Now create target columns
    merged_df["points_next_three"] = (
        merged_df.groupby(["season", "name"], group_keys=False)
        .apply(lambda g: rolling_forward_sum(g, 3))
    )
    merged_df["points_next_five"] = (
        merged_df.groupby(["season", "name"], group_keys=False)
        .apply(lambda g: rolling_forward_sum(g, 5))
    )
    print("Added target columns: points_next_three and points_next_five.")

    # Save the merged data to a new CSV file
    merged_df.to_csv(DATA_DIR / "merged_data.csv", index=False)
    print(f"Merged data saved to {DATA_DIR / 'merged_data.csv'}")
