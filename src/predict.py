import json
import sys
from pathlib import Path

import joblib
import pandas as pd

# Add src folder to sys.path
SRC_DIR = Path(__file__).resolve().parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from constants import DATA_DIR


def load_latest_features() -> pd.DataFrame:
    """Load feature-engineered live data produced by features.py."""
    data_path = DATA_DIR / "data_with_features.csv"
    if not data_path.exists():
        raise FileNotFoundError(
            f"Could not find {data_path}. Run features.py first"
        )
    return pd.read_csv(data_path)


def generate_predictions():
    model_path = DATA_DIR / "models" / "ridge_model.joblib"
    config_path = DATA_DIR / "all_feature_names.json"

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
        top_picks = pos_df.sort_values(
            by="predicted_3gw_pts", ascending=False
        ).head(5)

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