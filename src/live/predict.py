import json
import sys
from pathlib import Path

import joblib
import pandas as pd

# Put the repo root on sys.path so we can resolve `from src import ...` properly
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.common.constants import MODELS_DIR


def load_latest_features() -> pd.DataFrame:
    """Load feature-engineered live data, ready for the model to predict on.

    TODO(live features): this doesn't exist yet, and needs more than just
    reading a CSV. src.common.features.Features.transform() computes rolling
    "last 1/3/5 gameweek" windows and a points EMA per player - it needs
    per-gameweek history to do that. src.live.get_live_data.py currently only
    calls the FPL bootstrap-static endpoint, which returns one row per player
    with season-to-date CUMULATIVE totals (e.g. total minutes so far), not
    per-gameweek rows. Feeding that snapshot straight into Features.transform()
    would produce rolling windows that don't make sense (or fall back to all-zeros).

    To implement this properly:
      1. Fetch per-gameweek history for the current season for every player,
         e.g. FPL's `element-summary/{player_id}/` endpoint (returns each
         player's gameweek-by-gameweek history), or maintain our own log of
         daily/weekly snapshots as the season progresses.
      2. Reshape that into the same one-row-per-player-per-gameweek shape as
         data/merged_data.csv (same column names Features expects).
      3. Run it through src.common.features.Features.transform() - the exact
         code path used to build training features - so live and training
         features are computed identically and don't quietly drift apart
         (train/serve skew).
      4. Save the result (e.g. data/live/data_with_features.csv) and load it
         here instead of raising.

    Until then this deliberately fails loudly rather than silently predicting
    on the training feature set (data/data_with_features.csv), which would
    look like it worked but wouldn't be live data at all.
    """
    raise NotImplementedError(
        "Live feature building isn't implemented yet - see the docstring on "
        "load_latest_features() for what's missing and why."
    )


def generate_predictions():
    model_path = MODELS_DIR / "ridge_model.joblib"
    config_path = MODELS_DIR / "all_feature_names.json"

    if not model_path.exists():
        raise FileNotFoundError(
            f"Model file not found at {model_path}. Run train_model.py first"
        )

    # Load serialised pipeline and feature list
    print(f"Loading trained Ridge model from: {model_path}")
    model = joblib.load(model_path)

    with open(config_path, "r") as f:
        feature_cols = json.load(f)["feature_columns"]

    # Load latest player data
    df = load_latest_features()

    # Drop rows with missing features and run .predict()
    clean_df = df.dropna(subset=feature_cols).copy()
    X_live = clean_df[feature_cols]

    print("Generating predictions using model.predict()...")
    clean_df["predicted_3gw_pts"] = model.predict(X_live)

    # Display top picks by position
    pos_col = "position" if "position" in clean_df.columns else "element_type"
    positions = ["GKP", "DEF", "MID", "FWD"]

    print("\n" + "=" * 60)
    print("FPL ORACLE: TOP PREDICTED PICKS (NEXT 3 GAMEWEEKS)")
    print("=" * 60)

    for pos in positions:
        pos_df = clean_df[clean_df[pos_col] == pos]
        top_picks = pos_df.sort_values(by="predicted_3gw_pts", ascending=False).head(5)

        print(f"\n--- {pos} ---")
        for _, row in top_picks.iterrows():
            name = row.get("web_name", row.get("name", "Unknown"))
            team = row.get("team_name", row.get("team", ""))
            cost = row.get("now_cost", 0) / 10 if "now_cost" in row else 0.0
            pts = row["predicted_3gw_pts"]
            print(f"  • {name:<20} ({team:<3}) | £{cost:<4.1f}m | {pts:.2f} pts")

    print("\n" + "=" * 60 + "\n")


if __name__ == "__main__":
    generate_predictions()
