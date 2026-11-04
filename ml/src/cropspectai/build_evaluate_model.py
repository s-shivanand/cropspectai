"""Tune, save, and evaluate an XGBoost model on held-out test data.

This entry point combines data preparation and reusable modeling helpers.
It consumes prepared feature files; it does not engineer features itself.
Before running:
    - Set TARGET to the label to predict, such as "cab".
    - Set TRAIN_PATH and TEST_PATH to separate prepared Parquet datasets.
      Both must contain the selected channel/feature columns and TARGET.
    - Use DROP_CHANNELS to exclude sensor channels and EXTRA_FEATURES to
      select already-computed features. Use [] for channels only.
    - Choose a new MODEL_VERSION to avoid overwriting existing outputs.
      MODEL_PATH and RESULTS_PATH are derived automatically.

Run from the repository root:
    uv run python -m cropspectai.build_evaluate_model
"""

import json
from pathlib import Path
from typing import Literal

import joblib
import pandas as pd
from xgboost import XGBRegressor

from cropspectai import ml_config
from cropspectai.data_analysis import preprocessing
from cropspectai.modeling.evaluate import evaluate_model
from cropspectai.modeling.train import build_pipeline, tune_xgboost

#--------------------------------- Settings ------------------------------------------
TARGET: str = "cab"
TRAIN_PATH: Path = (
    ml_config.PROCESSED_PARQUET_DATA_SPLITS_PATH / TARGET / "train_all_features.parquet"
)
TEST_PATH: Path = (
    ml_config.PROCESSED_PARQUET_DATA_SPLITS_PATH / TARGET / "test" / "test_all_features.parquet"
)

# Channels to exclude; for example: ["F2", "FZ"]. Leave empty to use all channels.
DROP_CHANNELS: list[str] = []

# Engineered features to include. Add one per line, or comment out to exclude.
EXTRA_FEATURES: list[str] = [
    'mndvi_as7343',
    'sipi_as7343',
    'ci_red_edge_as7343',
    'psri_as7343',
    'red_green_nd',
    'datt1_as7343',
    'mcari_osavi_as7343'
]

N_TRIALS: int = 50  # Configurations to try; more trials take longer.
N_SPLITS: int = 7  # Cross-validation folds; use 2 or more, up to the training sample count.
# "r2" maximizes R-squared; "neg_mean_absolute_percentage_error" minimizes MAPE.
SCORING: Literal["r2", "neg_mean_absolute_percentage_error"] = "r2"
RANDOM_STATE: int = 42  # Keep fixed to reproduce fold splits and tuning suggestions.

# Change the version before running to keep previous models.
MODEL_VERSION: str = "v_01"
MODEL_PATH: Path = ml_config.MODELS_DIR / f"xgb_{MODEL_VERSION}.joblib"
RESULTS_PATH: Path = ml_config.MODELS_DIR / f"xgb_best_parameters_{MODEL_VERSION}.json"

#-------------------------------------------------------------------------------------

FEATURES: list[str] = [
    channel for channel in ml_config.AS7343_CHANNELS if channel not in DROP_CHANNELS
]
FEATURES += EXTRA_FEATURES


def evaluate_test_data(model_path: str | Path) -> tuple[dict[str, float], int]:
    """Evaluate a saved pipeline on the configured held-out test dataset.

    Args:
        model_path: Path to the pipeline serialized with Joblib.

    Returns:
        Pair of the test metrics dictionary and the number of labeled test
        samples evaluated. Metrics contain MAE, MAPE in percent, RMSE, and
        R-squared, with values converted to Python floats.

    Raises:
        FileNotFoundError: The configured test dataset or model is missing.
        KeyError: A selected feature or the target is absent from the dataset.
        ValueError: Fewer than two labeled samples remain after cleaning,
            or the pipeline rejects the features or targets.

    Note:
        Uses TEST_PATH, FEATURES, and TARGET from this module. Feature
        selection precedes deduplication, and feature order matches training.
        Prints the results without refitting the saved pipeline.
    """
    df = pd.read_parquet(TEST_PATH)
    df = df[FEATURES + [TARGET]].copy()
    df = preprocessing.drop_duplicates(df, TARGET)
    X, y = preprocessing.split_dataset_into_X_y(df, TARGET)

    if len(X) < 2:
        raise ValueError("Test data must contain at least two labeled samples for R².")

    # Only predict: imputation and model fitting were completed on training data.
    metrics = evaluate_model(X, y[TARGET], model_path)
    metrics = {name: float(value) for name, value in metrics.items()}
    print(f"Test results ({len(X)} samples):")
    print(json.dumps(metrics, indent=2, ensure_ascii=False))
    return metrics, len(X)


def main() -> None:
    """Tune, refit, save, and evaluate the configured XGBoost pipeline.

    Reads TRAIN_PATH, selects FEATURES and TARGET, and removes duplicate or
    unlabeled rows. Tuning fits preprocessing within each validation fold.
    The best configuration is then refitted on all training samples with
    mean imputation and no feature scaling, saved, and evaluated on TEST_PATH.

    Raises:
        FileNotFoundError: A configured input dataset is missing.
        KeyError: A selected feature or target is absent from an input dataset.
        ValueError: Training, tuning, or held-out evaluation fails validation.
        OSError: An output directory or artifact cannot be written.

    Note:
        Settings are module-level constants. Writes the fitted pipeline to
        MODEL_PATH and parameters, feature order, sample counts, validation
        scores, and test metrics to RESULTS_PATH. Existing artifacts at
        those paths are overwritten. Prints progress and output paths.
    """
    df = pd.read_parquet(TRAIN_PATH)

    # Select first: rows may become duplicates when unused features are removed.
    df = df[FEATURES + [TARGET]].copy()
    df = preprocessing.drop_duplicates(df, TARGET)
    X, y = preprocessing.split_dataset_into_X_y(df, TARGET)
    y = y[TARGET]

    print(f"Training {TARGET} with {len(X)} samples and {len(FEATURES)} features")
    print(f"Features: {FEATURES}")

    # The helper fits imputation inside each CV fold to avoid data leakage.
    # XGBoost does not need feature scaling.
    study = tune_xgboost(
        X,
        y,
        n_splits=N_SPLITS,
        n_trials=N_TRIALS,
        scale=False,
        scoring=SCORING,
        random_state=RANDOM_STATE,
    )

    best_trial = study.best_trial
    best_params = best_trial.user_attrs["model_params"]

    # Refit the winning pipeline on all training data before saving.
    pipeline = build_pipeline(XGBRegressor(**best_params), scale=False)
    pipeline.fit(X, y)

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, MODEL_PATH)
    test_metrics, n_test_samples = evaluate_test_data(MODEL_PATH)

    results = {
        "target": TARGET,
        "scoring": SCORING,
        "score_units": ("percent" if SCORING == "neg_mean_absolute_percentage_error" else "R2"),
        "mean_cv_score": study.best_value,
        "std_cv_score": best_trial.user_attrs["score_std"],
        "fold_scores": best_trial.user_attrs["fold_scores"],
        "best_params": best_params,
        "features": X.columns.tolist(),
        "n_samples": len(X),
        "test_metrics": test_metrics,
        "n_test_samples": n_test_samples,
        "test_data_path": str(TEST_PATH),
    }

    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with RESULTS_PATH.open("w") as file:
        json.dump(results, file, indent=2)

    print(f"Saved model to {MODEL_PATH}")
    print(f"Saved parameters, features, and test results to {RESULTS_PATH}")


if __name__ == "__main__":
    main()
