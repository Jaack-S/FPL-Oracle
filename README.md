# FPL-Oracle

A place to do FPL data analysis and predict player points. Soon, we will be using the model to predict future points.

## Training a new model

Run in order, from the project root:

1. `python getdata.py` — downloads historical per-season gameweek data into `data/raw/vaastav/`.
2. `python src/merge_data.py` — cleans and concatenates all seasons, adds target columns (`points_next_three`, `points_next_five`), writes `data/merged_data.csv`.
3. `python src/features.py` — builds rolling/EMA/opponent-strength features from `merged_data.csv`, writes `data/data_with_features.csv` and `data/all_feature_names.json`.
4. `python train_model.py` — tunes Ridge `alpha` via seasonal cross-validation, fits the final model, saves `data/models/ridge_model.joblib`.
5. `model_evaluation.ipynb` (optional) — loads the trained model, compares it against the `MovingAverageModel` baseline using `src/evaluate.py` metrics and `src/visualise.py` plots.

## Live inference (in progress, not yet wired up)

- **`fetch_data.py`** — hits the live FPL `bootstrap-static` API and saves current player data (with position/team names joined in) to `live_fpl_data.csv`.
- **`src/predict.py`** — loads the saved Ridge model and feature list, then generates and prints top-5 predicted picks per position.

**Gap:** `predict.py` currently loads `data/data_with_features.csv` (the training feature set) rather than the output of `fetch_data.py`. Nothing yet runs `live_fpl_data.csv` through `Features.transform()` to build live features, and there's no scheduler (cron/GitHub Actions) triggering a daily run. The live path needs those two pieces before it produces real day-of predictions.
