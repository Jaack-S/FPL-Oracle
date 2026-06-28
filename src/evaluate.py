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

class SeasonMetrics:
    pass

class AllMetrics:
    pass

def compute_metrics(df):
    return {
        "mean_absolute_error": 0,
        "rank_correlation": 0,
    }
