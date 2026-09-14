from src.common.constants import DATA_DIR, MODELS_DIR
import pandas as pd
import joblib
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


# Global Constants
POSITIONS = ["GKP", "DEF", "MID", "FWD"]
TOP_N_PER_POSITION = 5


def load_latest_features() -> pd.DataFrame:
    """Load feature-engineered live data, ready for the model to predict on.

    Built by src.live.get_live_data, which fetches per-gameweek player history
    from the FPL API and runs it through the same Features.transform() code
    path used for training.
    """
    features_path = DATA_DIR / "live/data_with_features.csv"

    if not features_path.exists():
        raise FileNotFoundError(
            f"Live feature data not found at {features_path}. Run "
            "src/live/get_live_data.py first."
        )

    return pd.read_csv(features_path)


def predict_points(model, feature_cols: list[str]) -> pd.DataFrame:
    """Load live features and score every player with predicted_3gw_pts."""
    df = load_latest_features()

    clean_df = df.dropna(subset=feature_cols).copy()
    X_live = clean_df[feature_cols]
    clean_df["predicted_3gw_pts"] = model.predict(X_live)

    return clean_df


def top_picks_by_position(
    clean_df: pd.DataFrame, top_n: int = TOP_N_PER_POSITION
) -> dict[str, pd.DataFrame]:
    pos_col = "position" if "position" in clean_df.columns else "element_type"

    return {
        pos: clean_df[clean_df[pos_col] == pos]
        .sort_values(by="predicted_3gw_pts", ascending=False)
        .head(top_n)
        for pos in POSITIONS
    }


def format_top_picks(top_picks: dict[str, pd.DataFrame]) -> str:
    lines = [
        "=" * 60, "FPL ORACLE: TOP PREDICTED PICKS (NEXT 3 GAMEWEEKS)", "=" * 60]

    for pos, pos_df in top_picks.items():
        lines.append(f"\n--- {pos} ---")
        for _, row in pos_df.iterrows():
            name = row.get("web_name", row.get("name", "Unknown"))
            team = row.get("team_name", row.get("team", ""))
            cost = row.get("now_cost", row.get("value", 0)) / 10
            pts = row["predicted_3gw_pts"]
            lines.append(
                f"  • {name:<20} ({team:<3}) | £{cost:<4.1f}m | {pts:.2f} pts")

    lines.append("\n" + "=" * 60)
    return "\n".join(lines)


def generate_predictions() -> str:
    model_path = MODELS_DIR / "ridge_model.joblib"
    config_path = MODELS_DIR / "all_feature_names.json"

    if not model_path.exists():
        raise FileNotFoundError(
            f"Model file not found at {model_path}. Run train_model.py first"
        )

    model = joblib.load(model_path)

    with open(config_path, "r") as f:
        feature_cols = json.load(f)["feature_columns"]

    clean_df = predict_points(model, feature_cols)

    message = format_top_picks(top_picks_by_position(clean_df))
    print(message)
    return message


if __name__ == "__main__":
    generate_predictions()
