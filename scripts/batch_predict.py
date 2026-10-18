"""Predict cab for a reflectance CSV: python scripts/batch_predict.py."""

import json
import math
import sys

import pandas as pd

from cropspectai import config
from cropspectai.data_analysis.feature_engineering import add_all_features
from cropspectai.modeling.predict import predict_model

FIRMWARE_SRC = config.PROJ_ROOT / "firmware" / "esp32-as7343-spectrometer" / "src"
sys.path.insert(0, str(FIRMWARE_SRC))
import channel_config  # noqa: E402

# Settings

#model
MODEL_VERSION = "v_01"
MODEL_PATH = config.MODELS_DIR / f"xgb_{MODEL_VERSION}.joblib"
MODEL_CONFIG_PATH = config.MODELS_DIR / f"xgb_best_parameters_{MODEL_VERSION}.json"

#trail data
CSV_FILE = channel_config.MISC_DATA_DIR / "reflectance.csv" # input
PREDICTION_FILE = config.PREDICTION_DATA_DIR / "misc" / f"batch_prediction_{MODEL_VERSION}.csv" # output

# #basil experiment data
# CSV_FILE = channel_config.RAW_DATA_DIR / "basil_experiment.csv" # input
# PREDICTION_FILE = config.BASIL_EXPERIMENT_DATA_PATH / f"batch_prediction_{MODEL_VERSION}.csv" # output

CHANNELS = list(config.AS7343_CHANNELS)
MEASUREMENT_COLUMNS = ["timestamp", "specimen", *CHANNELS, "noisy_channels"]


def main():
    """Predict every input row and replace the output CSV with the results."""
    if CSV_FILE.resolve() == PREDICTION_FILE.resolve():
        raise ValueError("Input and prediction CSV paths must be different")

    measurements = pd.read_csv(CSV_FILE, keep_default_na=False)
    missing = set(MEASUREMENT_COLUMNS) - set(measurements.columns)
    if missing:
        raise ValueError(f"Measurement CSV is missing columns: {sorted(missing)}")

    with MODEL_CONFIG_PATH.open() as handle:
        features = json.load(handle)["features"]

    output = measurements[MEASUREMENT_COLUMNS].copy()
    X = measurements[CHANNELS].apply(pd.to_numeric, errors="raise")
    X = add_all_features(X)[features]
    for feature in features:
        output[feature] = X[feature]

    if not measurements.empty:
        predictions = predict_model(X, MODEL_PATH)
        if not all(math.isfinite(value) for value in predictions):
            raise ValueError("Model returned a non-finite cab value")
        output["cab"] = predictions
    else:
        output["cab"] = pd.Series(dtype=float)

    PREDICTION_FILE.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(PREDICTION_FILE, index=False)
    # print(output[["timestamp", "specimen", "cab"]].to_string(index=False))
    print(f"Saved {len(output)} predictions to {PREDICTION_FILE}")


if __name__ == "__main__":
    main()
