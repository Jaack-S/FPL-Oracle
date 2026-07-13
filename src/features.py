# Creating Features

import os
import pandas as pd

class Features:
    def __init__(self, windows=[1, 3, 5], min_periods: int = 1):
        self.windows = windows
        self.min_periods = min_periods
        self.metrics = ["total_points", "minutes", "expected_goals", "expected_assists", "expected_goals_conceded", "bonus"]

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        df = X.sort_values(["name", "season", "GW"])

        # Make sure team columns are strings
        if "team" in df.columns:
            df["team"] = df["team"].astype(str)
        if "opponent_team" in df.columns:
            df["opponent_team"] = df["opponent_team"].astype(str)

        # Home/Away feature
        if "was_home" in df.columns:
            df["is_home"] = df["was_home"].astype(bool).astype(int)
        elif "is_home" in df.columns:
            df["is_home"] = df["is_home"].astype(int)

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

        # EMA feature to weight predictions on more recent gameweeks in the 3 or 5 week average (can adjust span)
        if "total_points" in df.columns:
            df["points_ema_3"] = (
                grouped["total_points"]
                .shift(1)
                .groupby(df["name"] + "_" + df["season"].astype(str))
                .transform(lambda x: x.ewm(span=3, adjust=False).mean())
            )
            df["points_ema_3"] = df["points_ema_3"].fillna(0)
        
        # Feature for opponent's xGC to find "weaker" defences
        if "team" in df.columns and "expected_goals_conceded" in df.columns and "GW" in df.columns:
            team_xgc = (df.groupby(["team", "season", "GW"])["expected_goals_conceded"]
                .mean()
                .reset_index())
            team_xgc = team_xgc.sort_values(["team", "season", "GW"])

            # Shift by 1
            team_xgc["team_xGC_last_3"] = (team_xgc.groupby(["team", "season"])["expected_goals_conceded"]
                .shift(1)
                .groupby(team_xgc["team"] + "_" + team_xgc["season"].astype(str))
                .rolling(window=3, min_periods=1)
                .sum()
                .reset_index(level=0, drop=True))

            # Map this back to the main dataframe based on who the opponent is
            if "opponent_team" in df.columns:
                df = df.merge(team_xgc[
                        ["team", "season", "GW", "team_xGC_last_3"]
                    ].rename(columns={"team": "opponent_team"}),
                    on=["opponent_team", "season", "GW"],
                    how="left",)
                df["team_xGC_last_3"] = df["team_xGC_last_3"].fillna(0)

        # One-hot encode for position
        position_col = ("position" if "position" in df.columns else "element_type")

        if position_col in df.columns:
            df = pd.get_dummies(df, columns=[position_col], dtype=int)
        return df

def main():
    input_filename = "merged_data.csv"
    output_filename = "data_with_features.csv"

    if not os.path.exists(input_filename):
        print(f"Error: {input_filename} not found.")
        return

    df = pd.read_csv(input_filename, low_memory = False)

    engineer = Features(windows=[1, 3, 5])
    df_features = engineer.transform(df)

    df_features.to_csv(output_filename, index=False)

if __name__ == "__main__":
    main()