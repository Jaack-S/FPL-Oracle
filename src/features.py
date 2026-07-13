# Creating Features

import json

import pandas as pd

from constants import DATA_DIR

POSITION_MAP = {1: "GK", 2: "DEF", 3: "MID", 4: "FWD"}


class Features:
    POSITIONS = ["GK", "DEF", "MID", "FWD"]

    def __init__(self, windows=[1, 3, 5], min_periods: int = 1):
        self.windows = windows
        self.min_periods = min_periods
        # NOTE: expected_goals/expected_assists/expected_goals_conceded are entirely NaN in the data
        # before 2022-23, and the rolling computation below fillna(0)s them - this means that the
        # _last_1/3/5 columns for these three are faulty until backfilled from another source (e.g. Understat).
        self.metrics = ["total_points", "minutes", "expected_goals", "expected_assists", "expected_goals_conceded", "bonus"]
        self.flat_features = ["was_home", "value"]

    @staticmethod
    def _group_season_key(names: pd.Series, seasons: pd.Series) -> pd.Series:
        return names.astype(str) + "_" + seasons.astype(str)

    def _available_metrics(self, df: pd.DataFrame) -> list[str]:
        return [col for col in self.metrics if col in df.columns]

    def _encode_position(self, df: pd.DataFrame) -> pd.DataFrame:
        position_col = "position" if "position" in df.columns else "element_type"
        if position_col not in df.columns:
            return df

        if position_col == "element_type":
            positions = df["element_type"].map(POSITION_MAP)
        else:
            positions = df["position"]

        position = pd.Categorical(positions, categories=self.POSITIONS)
        dummies = pd.get_dummies(position, prefix="position").astype(int)
        return pd.concat([df, dummies], axis=1)

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        df = X.sort_values(["name", "season", "GW"])

        # Make sure team columns are strings
        if "team" in df.columns:
            df["team"] = df["team"].astype(str)
        if "opponent_team" in df.columns:
            df["opponent_team"] = df["opponent_team"].astype(str)

        # Home/Away feature
        if "was_home" in df.columns:
            df["was_home"] = df["was_home"].astype(int)

        available_metrics = self._available_metrics(df)
        missing = set(self.metrics) - set(available_metrics)
        if missing:
            print(f"Missing columns: {missing}")

        grouped = df.groupby(["name", "season"])
        player_key = self._group_season_key(df["name"], df["season"])

        for col in available_metrics:
            shift_col = grouped[col].shift(1)
            for w in self.windows:
                feature_name = f"{col}_last_{w}"
                df[feature_name] = (
                    shift_col.groupby(player_key)
                    .rolling(window=w, min_periods=self.min_periods)
                    .sum()
                    .reset_index(level=0, drop=True)
                )
                df[feature_name] = df[feature_name].fillna(0)

        # EMA feature to weight predictions on more recent gameweeks in the 3 or 5 week average (can adjust span)
        if "total_points" in df.columns:
            df["points_ema_3"] = (
                grouped["total_points"]
                .shift(1)
                .groupby(player_key)
                .transform(lambda x: x.ewm(span=3, adjust=False).mean())
                .fillna(0)
            )

        # Feature for opponent's xGC to find "weaker" defences
        if "team" in df.columns and "expected_goals_conceded" in df.columns and "GW" in df.columns:
            team_xgc = (
                df.groupby(["team", "season", "GW"])["expected_goals_conceded"]
                .mean()
                .reset_index()
                .sort_values(["team", "season", "GW"])
            )
            team_key = self._group_season_key(team_xgc["team"], team_xgc["season"])
            team_xgc["team_xGC_last_3"] = (
                team_xgc.groupby(["team", "season"])["expected_goals_conceded"]
                .shift(1)
                .groupby(team_key)
                .rolling(window=3, min_periods=1)
                .sum()
                .reset_index(level=0, drop=True)
            )
            team_xgc["team_xGC_last_3"] = team_xgc["team_xGC_last_3"].fillna(0)

            if "opponent_team" in df.columns:
                df = df.merge(
                    team_xgc[["team", "season", "GW", "team_xGC_last_3"]].rename(columns={"team": "opponent_team"}),
                    on=["opponent_team", "season", "GW"],
                    how="left",
                )
                df["team_xGC_last_3"] = df["team_xGC_last_3"].fillna(0)

        return self._encode_position(df)

    def get_feature_names(self, df: pd.DataFrame) -> list[str]:
        """Return list of feature column names created by transform."""
        feature_cols = []

        for col in self._available_metrics(df):
            for w in self.windows:
                feature_name = f"{col}_last_{w}"
                if feature_name in df.columns:
                    feature_cols.append(feature_name)

        feature_cols += [col for col in self.flat_features if col in df.columns]
        feature_cols += [f"position_{p}" for p in self.POSITIONS if f"position_{p}" in df.columns]

        return feature_cols

    def get_unscaled_features(self, df: pd.DataFrame) -> list[str]:
        return [f"position_{p}" for p in self.POSITIONS if f"position_{p}" in df.columns]


def main():
    input_path = DATA_DIR / "merged_data.csv"
    output_path = DATA_DIR / "data_with_features.csv"
    feature_config_path = DATA_DIR / "all_feature_names.json"

    if not input_path.exists():
        raise FileNotFoundError(f"File not found: {input_path}")

    df = pd.read_csv(input_path)

    engineer = Features(windows=[1, 3, 5])
    df_features = engineer.transform(df)

    df_features.to_csv(output_path, index=False)
    print(f"Feature data saved to {output_path}")

    # Export feature names and metadata
    feature_names = engineer.get_feature_names(df_features)
    feature_config = {
        "feature_columns": feature_names,
        "windows": engineer.windows,
        "metrics": engineer.metrics,
        "min_periods": engineer.min_periods,
        "flat_features": engineer.flat_features,
        "unscaled_features": engineer.get_unscaled_features(df_features),
        "positions": engineer.POSITIONS,
    }

    with open(feature_config_path, "w") as f:
        json.dump(feature_config, f, indent=2)

    print(f"Feature config saved to {feature_config_path}")
    print(f"Total features created: {len(feature_names)}")


if __name__ == "__main__":
    main()
