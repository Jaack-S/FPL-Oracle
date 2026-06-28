import pandas as pd


class MovingAverageModel:
    def __init__(self, window: int = 3, min_periods: int = 1):
        self.window = window
        self.min_periods = min_periods

    def predict(self, X: pd.DataFrame) -> pd.DataFrame:
        df = X.sort_values(["name", "season", "GW"])

        # Shift by 1 so GW t uses only history up to GW t-1
        df["points_lag1"] = df.groupby(["name", "season"])["total_points"].shift(1)

        df["prediction_next_one"] = df.groupby(["name", "season"])[
            "points_lag1"
        ].transform(
            lambda x: x.rolling(self.window, min_periods=self.min_periods).mean()
        )
        df["prediction_next_three"] = df["prediction_next_one"] * 3
        df["prediction_next_five"] = df["prediction_next_one"] * 5

        return df
