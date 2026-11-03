"""Tune, save, and evaluate an XGBoost model on held-out test data.

Edit the settings below, then run from the repository root:
    python -m cropspectai.build_evaluate_model
"""

import json

import joblib
import pandas as pd
from xgboost import XGBRegressor

from cropspectai import ml_config
from cropspectai.data_analysis import preprocessing
from cropspectai.modeling.evaluate import evaluate_model
from cropspectai.modeling.train import build_pipeline, tune_xgboost

# Settings
TARGET = "cab"
TRAIN_PATH = ml_config.PROCESSED_PARQUET_DATA_SPLITS_PATH / TARGET / "train_all_features.parquet"
TEST_PATH = (
    ml_config.PROCESSED_PARQUET_DATA_SPLITS_PATH / TARGET / "test" / "test_all_features.parquet"
)

# Channels to exclude; for example: ["F2", "FZ"]. Leave empty to use all channels.
DROP_CHANNELS = []

# Engineered features to include. Add one per line, or comment out to exclude.
EXTRA_FEATURES = [
                  'mndvi_as7343', 
                #   'sipi_as7343', 
                #   'ci_red_edge_as7343', 
                #   'psri_as7343', 
                #   'red_green_nd', 
                #   'datt1_as7343', 
                #   'mcari_osavi_as7343'
                  ]

FEATURES = [channel for channel in ml_config.AS7343_CHANNELS if channel not in DROP_CHANNELS]
FEATURES += EXTRA_FEATURES

N_TRIALS = 50
N_SPLITS = 7
SCORING = "r2"  # "r2" or "neg_mean_absolute_percentage_error"
RANDOM_STATE = 42

# Change the version before running to keep previous models.
MODEL_VERSION = "v_05"
MODEL_PATH = ml_config.MODELS_DIR / f"xgb_{MODEL_VERSION}.joblib"
RESULTS_PATH = ml_config.MODELS_DIR / f"xgb_best_parameters_{MODEL_VERSION}.json"


def evaluate_test_data(model_path):
    """Evaluate the saved pipeline using the same features in the same order."""
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


def main():
    """Tune on training data, save the model, and record held-out test metrics."""
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
