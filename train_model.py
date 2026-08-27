"""
In this file we perform train/test splitting (across seasons) and model training.
During model training, we use cross-validation to find the best hyperparameters. Since
we are currently only using Ridge Regression (a regularised linear model), we only tune
the alpha (regularisation strength) parameter.
We then check for signs of overfitting by checking generalisation gap between train and test.
"""

import json
from pathlib import Path

import sys
import joblib
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


ROOT_DIR = Path(__file__).resolve().parent
if (ROOT_DIR / "src").exists():
    sys.path.insert(0, str(ROOT_DIR / "src"))
else:
    sys.path.insert(0, str(ROOT_DIR))

def load_feature_config(feature_config_path: Path) -> dict:
    with open(feature_config_path, "r") as f:
        return json.load(f)


def load_feature_names(feature_config_path: Path) -> list[str]:
    """
    Here we load _all_ feature column names from a JSON config file that
    is produced in features.py.
    TODO: we might want to perform feature selection to only include those that are
    actually useful in the model. At the moment it's fine, but eventually we
    may have loads of features and trimming the will improve model generalisation.
    """
    return load_feature_config(feature_config_path)["feature_columns"]


def _default_feature_config_path() -> Path:
    from src.constants import DATA_DIR
    return DATA_DIR / "all_feature_names.json"


def _resolve_feature_scaling(
    feature_cols: list[str] | None,
    unscaled_features: list[str] | None,
    feature_config_path: Path | None = None,
) -> tuple[list[str], list[str]]:
    if feature_cols is not None and unscaled_features is not None:
        return feature_cols, unscaled_features

    config = load_feature_config(feature_config_path or _default_feature_config_path())
    if feature_cols is None:
        feature_cols = config["feature_columns"]
    if unscaled_features is None:
        unscaled_features = config.get("unscaled_features", [])
    return feature_cols, unscaled_features


def scaling_pipeline(
    feature_cols: list[str],
    unscaled_features: list[str],
    model,
) -> Pipeline:
    scaled_cols = [col for col in feature_cols if col not in unscaled_features]
    unscaled_cols = [col for col in feature_cols if col in unscaled_features]

    if not unscaled_cols:
        return Pipeline([
            ("scaler", StandardScaler()),
            ("model", model),
        ])

    preprocessor = ColumnTransformer(
        [
            ("scaled", StandardScaler(), scaled_cols),
            ("unscaled", "passthrough", unscaled_cols),
        ],
        verbose_feature_names_out=False,
    )
    return Pipeline([
        ("scaler", preprocessor),
        ("model", model),
    ])


def seasonal_split(
    df: pd.DataFrame,
    test_seasons: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split by season. All seasons not in test_seasons are used for training.
    This is the best way to split time-series data — random splitting leaks
    future gameweeks into the training set.
    Example
    -------
    train_df, test_df = seasonal_split(df, test_seasons=["2023-24"])
    # trains on 2020-21, 2021-22, 2022-23 — tests blind on 2023-24
    """
    train = df[~df["season"].isin(test_seasons)].copy()
    test  = df[ df["season"].isin(test_seasons)].copy()
    return train, test


def train_and_predict(
    df: pd.DataFrame,
    model,
    target_col: str,
    test_seasons: list[str],
    feature_cols: list[str],
    prediction_col: str = "predicted_pts",
    verbose: bool = True,
) -> tuple[pd.DataFrame, dict]:
    """
    Train model on all seasons outside test_seasons, predict on test_seasons.
    Returns the full dataframe with a prediction column added for test rows,
    and a dictionary of performance metrics.
    Parameters
    ----------
    df            : dataframe with features already created (output of features.py)
    model         : any model interface that has fit / predict methods
    target_col    : target column name (e.g. "points_next_three")
    test_seasons  : seasons to hold out for evaluation
    feature_cols  : list of feature column names to use
    prediction_col: name of the column written back to df
    verbose       : if True, print train/test diagnostics
    """
    train_df, test_df = seasonal_split(df, test_seasons)

    # Drop rows where target or any feature is null
    train_clean = train_df.dropna(subset=feature_cols + [target_col])
    test_clean  = test_df.dropna(subset=feature_cols + [target_col])

    X_train = train_clean[feature_cols]
    y_train = train_clean[target_col]
    X_test  = test_clean[feature_cols]
    y_test  = test_clean[target_col]

    model.fit(X_train, y_train)

    # Get predictions on both train and test sets
    train_predictions = model.predict(X_train)
    test_predictions = model.predict(X_test)

    # Calculate metrics
    train_mae = mean_absolute_error(y_train, train_predictions)
    test_mae = mean_absolute_error(y_test, test_predictions)

    train_spearman, _ = spearmanr(y_train, train_predictions)
    test_spearman, _ = spearmanr(y_test, test_predictions)

    train_metrics = {
        "mae": train_mae,
        "spearman": train_spearman,
    }

    test_metrics = {
        "mae": test_mae,
        "spearman": test_spearman,
    }

    if verbose:
        print("\n" + "="*60)
        print("PERFORMANCE DIAGNOSTICS")
        print("="*60)
        print(f"\nTrain set ({len(train_clean)} samples):")
        print(f"  MAE:        {train_mae:.4f}")
        print(f"  Spearman R: {train_spearman:.4f}")

        print(f"\nTest set ({len(test_clean)} samples):")
        print(f"  MAE:        {test_mae:.4f}")
        print(f"  Spearman R: {test_spearman:.4f}")

        print("\nGeneralisation gap:")
        mae_gap = test_mae - train_mae
        mae_pct = (mae_gap / train_mae) * 100 if train_mae > 0 else 0
        spearman_gap = train_spearman - test_spearman
        spearman_pct = (spearman_gap / train_spearman) * 100 if train_spearman > 0 else 0

        print(f"  MAE gap:        {mae_gap:.4f} ({mae_pct:+.1f}%)")
        print(f"  Spearman gap:   {spearman_gap:+.4f} ({spearman_pct:+.1f}%)")

        print("\nOverfitting check:")
        if mae_gap > 1.0 or spearman_gap > 0.1:
            print("  ⚠️  Significant performance drop on test set - possible overfitting")
        elif mae_gap < -0.5 or spearman_gap < -0.05:
            print("  ⚠️  Test performance better than train - unusual, check for data leakage")
        else:
            print("  ✓  Reasonable generalization")
        print("="*60 + "\n")

    # Write predictions back to the original dataframe and return the metrics
    df = df.copy()
    df[prediction_col] = np.nan
    df.loc[test_clean.index, prediction_col] = test_predictions

    metrics = {
        "train": train_metrics,
        "test": test_metrics,
    }

    return df, metrics


# ---------------------------------------------------------------------------
# Hyperparameter tuning
# ---------------------------------------------------------------------------

def seasonal_cv_splits(df: pd.DataFrame, train_seasons: list[str], n_splits: int = 2):
    """
    Generate train/validation indices for seasonal cross-validation.
    Always trains on earlier seasons and validates on later seasons, respecting
    time ordering. For example with 7 training seasons and n_splits=3:
    - Fold 1: train on seasons 0-3, validate on season 4
    - Fold 2: train on seasons 0-4, validate on season 5
    - Fold 3: train on seasons 0-5, validate on season 6
    Yields (train_indices, val_indices) tuples compatible with GridSearchCV.
    """
    # Sort seasons chronologically
    sorted_seasons = sorted(train_seasons)

    if len(sorted_seasons) <= n_splits:
        raise ValueError(f"Need at least {n_splits + 1} seasons for {n_splits} CV splits, got {len(sorted_seasons)}")

    # Create folds: last n_splits seasons each become a validation set
    for i in range(n_splits):
        # Validation season is the (len - n_splits + i)th season
        val_season_idx = len(sorted_seasons) - n_splits + i
        val_season = sorted_seasons[val_season_idx]
        train_seasons_fold = sorted_seasons[:val_season_idx]

        # Get positional indices (not dataframe index values)
        train_mask = df["season"].isin(train_seasons_fold)
        val_mask = df["season"] == val_season

        train_indices = np.where(train_mask)[0]
        val_indices = np.where(val_mask)[0]

        yield train_indices, val_indices


def tune_ridge_alpha(
    df: pd.DataFrame,
    target_col: str,
    feature_cols: list[str],
    train_seasons: list[str],
    unscaled_features: list[str] | None = None,
    alphas: list[float] = None,
    cv_splits: int = 2,
) -> float:
    """
    Use seasonal cross-validation on training seasons to find best Ridge alpha value.
    Parameters
    ----------
    df            : dataframe with features
    target_col    : target column name
    feature_cols  : list of feature columns
    train_seasons : seasons to use for CV (should exclude test seasons)
    alphas        : list of alpha values to try
    cv_splits     : number of CV splits
    """
    if alphas is None:
        alphas = [0.01, 0.1, 0.5, 1.0, 5.0, 10.0, 50.0, 100.0]

    _, unscaled_features = _resolve_feature_scaling(feature_cols, unscaled_features)

    # Filter to training seasons only
    train_df = df[df["season"].isin(train_seasons)].copy()
    train_clean = train_df.dropna(subset=feature_cols + [target_col])

    X = train_clean[feature_cols]
    y = train_clean[target_col]

    pipeline = scaling_pipeline(feature_cols, unscaled_features, Ridge())

    # Generate seasonal CV folds
    cv_folds = list(seasonal_cv_splits(train_clean, train_seasons, n_splits=cv_splits))

    # Print CV structure for transparency
    sorted_seasons = sorted(train_seasons)
    print(f"\nSeasonal cross-validation ({cv_splits} folds):")
    for i, (train_idx, val_idx) in enumerate(cv_folds, 1):
        val_season_idx = len(sorted_seasons) - cv_splits + i - 1
        val_season = sorted_seasons[val_season_idx]
        train_seasons_fold = sorted_seasons[:val_season_idx]
        print(f"  Fold {i}: train on {train_seasons_fold[0]} to {train_seasons_fold[-1]}, validate on {val_season}")

    # Grid search with MAE as scoring metric (neg because sklearn maximizes)
    param_grid = {"model__alpha": alphas}
    grid_search = GridSearchCV(
        pipeline,
        param_grid,
        cv=cv_folds,
        scoring="neg_mean_absolute_error",
        n_jobs=-1,
    )

    grid_search.fit(X, y)

    best_alpha = grid_search.best_params_["model__alpha"]
    best_score = -grid_search.best_score_  # Convert back to positive MAE

    print(f"\n  Best alpha: {best_alpha}")
    print(f"  Best CV MAE: {best_score:.4f}")

    return best_alpha


def ridge_model(
    alpha: float = 1.0,
    feature_cols: list[str] | None = None,
    unscaled_features: list[str] | None = None,
    feature_config_path: Path | None = None,
) -> Pipeline:
    """
    Ridge regression (linear model with regularisation).
    Good first step after the moving average baseline:
    - Still interpretable — we can inspect model coefficients
    - The scaler is essential: rolling averages and was_home are on different scales
    - One-hot features (e.g. position dummies) are left unscaled
    - Regularisation (alpha) prevents overfitting on correlated rolling features
    Parameters
    ----------
    alpha : regularisation strength (higher = more regularisation)
    feature_cols : all feature columns, used to build the partial scaler
    unscaled_features : columns to pass through without scaling
    feature_config_path : optional path to feature config JSON
    """
    feature_cols, unscaled_features = _resolve_feature_scaling(
        feature_cols, unscaled_features, feature_config_path
    )
    return scaling_pipeline(feature_cols, unscaled_features, Ridge(alpha=alpha))


if __name__ == "__main__":
    from constants import DATA_DIR

    # Load data and feature configuration
    data_path = DATA_DIR / "data_with_features.csv"
    feature_config_path = DATA_DIR / "all_feature_names.json"

    print(f"Loading data from {data_path}")
    df = pd.read_csv(data_path)

    print(f"Loading feature configuration from {feature_config_path}")
    feature_config = load_feature_config(feature_config_path)
    feature_cols = feature_config["feature_columns"]
    unscaled_features = feature_config.get("unscaled_features", [])
    print(f"Using {len(feature_cols)} features")

    TEST_SEASONS = ["2024-25", "2025-26"]
    TRAIN_SEASONS = [s for s in df["season"].unique() if s not in TEST_SEASONS]
    TARGET = "points_next_three"

    # --- Ridge with hyperparameter tuning ---
    print("\n" + "="*60)
    print("RIDGE REGRESSION")
    print("="*60)

    best_alpha = tune_ridge_alpha(
        df=df,
        target_col=TARGET,
        feature_cols=feature_cols,
        train_seasons=TRAIN_SEASONS,
        unscaled_features=unscaled_features,
        cv_splits=2,
    )

    df_ridge, ridge_metrics = train_and_predict(
        df=df,
        model=ridge_model(alpha=best_alpha, feature_cols=feature_cols, unscaled_features=unscaled_features),
        target_col=TARGET,
        test_seasons=TEST_SEASONS,
        feature_cols=feature_cols,
        prediction_col="predicted_pts",
        verbose=True,
    )

    # Serialisation

    models_dir = DATA_DIR / "models"
    models_dir.mkdir(parents=True, exist_ok=True)
    model_save_path = models_dir / "ridge_model.joblib"

    print(
        f"Fitting final model with alpha={best_alpha} on all training"
        f" seasons ({len(TRAIN_SEASONS)} seasons)..."
    )
    # Filter to training seasons and clean NaNs
    train_df = df[df["season"].isin(TRAIN_SEASONS)].dropna(
        subset=feature_cols + [TARGET]
    )
    X_train = train_df[feature_cols]
    y_train = train_df[TARGET]

    final_model = ridge_model(
        alpha=best_alpha,
        feature_cols=feature_cols,
        unscaled_features=unscaled_features,
    )
    final_model.fit(X_train, y_train)

    joblib.dump(final_model, model_save_path)
    print(f"✅ Final pipeline successfully saved to: {model_save_path}")