"""
This file contains the code for visualising the metrics calculated in evaluate.py
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.training.evaluate import AllMetrics, compute_metrics


def plot_across_seasons():
    pass


def plot_within_season():
    pass

def plot_model_evaluation():
    """
    Chains together a bunch of metrics and plots a "master visualisation"

    1. plot_mae_by_season
    2. plot_mae_by_gameweek
    3. plot_predicted_vs_actual
    """
    pass


PALETTE = ["#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B2"]
POSITION_ORDER = ["GK", "DEF", "MID", "FWD"]
 
 
def _bar(ax, x_labels, values, title, ylabel, annotate=True):
    bars = ax.bar(x_labels, values, color="#4C72B0", edgecolor="white", width=0.6)
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.tick_params(axis="x", rotation=45)
    ax.grid(axis="y", alpha=0.3)
    if annotate:
        for bar, val in zip(bars, values):
            if val is not None and bar.get_height() is not None:
                height = bar.get_height()
                if pd.notna(height):
                    ax.text(
                        bar.get_x() + bar.get_width() / 2,
                        height + 0.005,
                        f"{val:.3f}", ha="center", va="bottom", fontsize=8,
                    )
 
 
def plot_mae_by_season(all_metrics: AllMetrics) -> plt.Figure:
    summary = all_metrics.summary()
    fig, ax = plt.subplots(figsize=(10, 4))
    _bar(ax, summary["season"], summary["mean_absolute_error"], "MAE by Season", "MAE (points)")
    plt.tight_layout()
    return fig
 
 
def plot_spearman_by_season(all_metrics: AllMetrics) -> plt.Figure:
    summary = all_metrics.summary()
    fig, ax = plt.subplots(figsize=(10, 4))
    _bar(ax, summary["season"], summary["rank_correlation"], "Rank Correlation by Season", "Spearman's r")
    ax.axhline(0, color="red", linewidth=0.8, linestyle="--")
    plt.tight_layout()
    return fig
 
 
def plot_mae_by_gameweek(all_metrics: AllMetrics) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(12, 4))
    for i, (season, sm) in enumerate(all_metrics.seasons.items()):
        gws = sorted(sm.by_gameweek.keys())
        maes = [sm.by_gameweek[gw]["mean_absolute_error"] or 0 for gw in gws]
        ax.plot(gws, maes, label=season, linewidth=1.8, color=PALETTE[i % len(PALETTE)])
    ax.set_xlabel("Gameweek")
    ax.set_ylabel("MAE (points)")
    ax.set_title("MAE by Gameweek")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    return fig
 
 
def plot_mae_by_position(all_metrics: AllMetrics) -> plt.Figure:
    seasons = list(all_metrics.seasons.keys())
    x = np.arange(len(POSITION_ORDER))
    width = 0.7 / len(seasons)
 
    fig, ax = plt.subplots(figsize=(10, 4))
    for i, (season, sm) in enumerate(all_metrics.seasons.items()):
        maes = [
            (sm.by_position.get(pos) or {}).get("mean_absolute_error") or 0
            for pos in POSITION_ORDER
        ]
        offset = (i - len(seasons) / 2) * width + width / 2
        ax.bar(x + offset, maes, width, label=season, color=PALETTE[i % len(PALETTE)])
    ax.set_xticks(x)
    ax.set_xticklabels(POSITION_ORDER)
    ax.set_ylabel("MAE (points)")
    ax.set_title("MAE by Position")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    return fig
 
 
def plot_predicted_vs_actual(
    df: pd.DataFrame,
    season: str,
    actual_col: str,
    predicted_col: str,
    sample_n: int = 2000,
) -> plt.Figure:
    s = df[df["season"] == season].dropna(subset=[actual_col, predicted_col])
    if len(s) > sample_n:
        s = s.sample(sample_n, random_state=42)
 
    metrics = compute_metrics(s, actual_col, predicted_col)
    max_val = max(s[predicted_col].max(), s[actual_col].max()) + 1
 
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(s[predicted_col], s[actual_col], alpha=0.2, s=10, color="#4C72B0")
    ax.plot([0, max_val], [0, max_val], "r--", linewidth=1.2, label="Perfect prediction")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(f"Predicted vs Actual — {season}")
    ax.legend()
    ax.text(
        0.05, 0.93,
        f"MAE={metrics['mean_absolute_error']}  ρ={metrics['rank_correlation']}",
        transform=ax.transAxes, fontsize=9,
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.7),
    )
    plt.tight_layout()
    return fig
 
 
def plot_all(
    all_metrics: AllMetrics,
    df: pd.DataFrame,
    actual_col: str,
    predicted_col: str,
) -> None:
    """
    Render all plots for a given set of metrics and a pandas DataFrame containing
    the raw predictions.
    """
    latest_season = sorted(all_metrics.seasons.keys())[-1]
 
    plot_mae_by_season(all_metrics)
    plt.show()
 
    plot_spearman_by_season(all_metrics)
    plt.show()
 
    plot_mae_by_gameweek(all_metrics)
    plt.show()
 
    plot_mae_by_position(all_metrics)
    plt.show()
 
    plot_predicted_vs_actual(df, latest_season, actual_col, predicted_col)
    plt.show()
