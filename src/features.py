# Creating Features

import os
import pandas as pd

class Features:
    def __init__(self, windows=[1, 3, 5], min_periods: int = 1):
        self.windows = windows
        self.min_periods = min_periods
        self.metrics = ["total_points", "minutes", "xG", "xA", "xGC", "bonus"]

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        df = X.sort_values(["name", "season", "GW"])

        available_metrics = [col for col in self.metrics if col in df.columns]
        missing = set(self.metrics) - set(available_metrics)

        if missing:
            print(f"Missing columns: {missing}")
        
        grouped = df.groupby(["name", "season"])

        for col in available_metrics:
            shift_col = grouped[col].shift(1)

            for w in self.windows:
                feature_name = f"{col}_last_{w}"

                df[feature_name] = (
                    shift_col.groupby(df["name"] + "_" + df["season"].astype(str))
                    .rolling(window=w, min_periods=self.min_periods)
                    .sum()
                    .reset_index(level=0, drop=True))

                df[feature_name] = df[feature_name].fillna(0)


        return df

def main():
    input_filename = "merged_data.csv"
    output_filename = "data_with_features.csv"

    if not os.path.exists(input_filename):
        print(f"Error: {input_filename} not found.")
        return

    df = pd.read_csv(input_filename)

    engineer = Features(windows=[1, 3, 5])
    df_features = engineer.transform(df)

    df_features.to_csv(output_filename, index=False)

if __name__ == "__main__":
    main()