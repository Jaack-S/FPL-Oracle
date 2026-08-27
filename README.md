# FPL-Oracle

A place to do FPL data analysis and predict player points. Soon, we will be using the model to predict future points.

## Layout

- `src/common/` — shared code used by both training and live inference: `constants.py` (paths, seasons, column definitions) and `features.py` (the `Features` class that builds rolling/EMA/opponent-strength features).
- `src/training/` — the offline pipeline: fetching historical data, cleaning, merging, feature building, model training, and evaluation.
- `src/live/` — daily live inference: fetching current FPL data and generating predictions from the trained model.

## Training a new model

Run in order, from the project root:

1. `python src/training/get_training_data.py` — downloads historical per-season gameweek data into `data/raw/vaastav/`.
2. `python src/training/merge_data.py` — cleans and concatenates all seasons, adds target columns (`points_next_three`, `points_next_five`), writes `data/merged_data.csv`.
3. `python src/common/features.py` — builds rolling/EMA/opponent-strength features from `merged_data.csv`, writes `data/data_with_features.csv` and `data/all_feature_names.json`.
4. `python src/training/train_model.py` — tunes Ridge `alpha` via seasonal cross-validation, fits the final model, saves `models/ridge_model.joblib`.
5. `model_evaluation.ipynb` (optional) — loads the trained model, compares it against the `MovingAverageModel` baseline using `src/training/evaluate.py` metrics and `src/training/visualise.py` plots.

## Live inference (in progress, not yet wired up)

- `python src/live/get_live_data.py` — hits the live FPL `bootstrap-static` API and saves current player data (with position/team names joined in) to `data/live/live_fpl_data.csv`.
- `python src/live/predict.py` — loads the saved Ridge model and feature list, then would generate and print top-5 predicted picks per position.

**Gap:** `predict.py` currently raises `NotImplementedError` — see the docstring for an explanation.