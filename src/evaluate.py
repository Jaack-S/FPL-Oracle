"""
This file contains the code for evaluating each of the models we write. It assumes
that every model creates 'prediction' columns that can be compared to the relevant target
column.

We calculate:
- Mean Absolute Error to see if we are accurately predicting future points
- Rank correlation (Spearman's R) to figure out if we are ordering the players by 'how highly
  will they score?' correctly

We'll calculate metrics per-gameweek and per-position, for each season.
"""

from dataclasses import dataclass, field

import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import mean_absolute_error


def compute_metrics(df: pd.DataFrame, actual_col: str, predicted_col: str) -> dict:
    # drop any rows where "actual points" or "predicted points" are null
    clean = df.dropna(subset=[actual_col, predicted_col])

    if len(clean) < 2:
        # Don't bother calculating metrics in this case
        return {"mean_absolute_error": None, "rank_correlation": None}
    
    # calculate spearman's correlation coefficient (rank based correlation)
    r, _ = spearmanr(clean[actual_col], clean[predicted_col])

    # return the metrics we care about
    return {
        "mean_absolute_error": round(mean_absolute_error(clean[actual_col], clean[predicted_col]), 3),
        "rank_correlation": round(r, 3),
    }


@dataclass
class SeasonMetrics:
    """Metrics for a single season, broken down by gameweek and position."""
    season: str
    overall: dict
    by_position: dict[str, dict] = field(default_factory=dict)
    by_gameweek: dict[int, dict] = field(default_factory=dict)

    @classmethod
    def from_dataframe(
        cls,
        df: pd.DataFrame,
        season: str,
        actual_col: str,
        predicted_col: str,
    ):
        season_df = df[df['season'] == season]

        overall = compute_metrics(season_df, actual_col, predicted_col)

        by_position = {
            pos: compute_metrics(pos_df, actual_col, predicted_col)
            for pos, pos_df in season_df.groupby('position')
        }

        by_gameweek = {
            int(gw): compute_metrics(gw_df, actual_col, predicted_col)
            for gw, gw_df in season_df.groupby('GW')
            if len(gw_df) > 0
        }

        return cls(season=season, overall=overall, by_position=by_position, by_gameweek=by_gameweek)


@dataclass
class AllMetrics:
    """Metrics across all seasons."""
    seasons: dict[str, SeasonMetrics] = field(default_factory=dict)

    @classmethod
    def from_dataframe(
        cls,
        df: pd.DataFrame,
        actual_col: str,
        predicted_col: str,
    ):
        seasons = {
            season: SeasonMetrics.from_dataframe(df, season, actual_col, predicted_col)
            for season in sorted(df['season'].unique())
        }
        return cls(seasons=seasons)

    def summary(self) -> pd.DataFrame:
        return pd.DataFrame([
            {'season': s, **m.overall}
            for s, m in self.seasons.items()
        ])
