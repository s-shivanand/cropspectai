"""Measure AS7343 reflectance and predict cab: python measure_predict.py."""

import csv
import json
import math
import sys

import pandas as pd

from cropspectai import config
from cropspectai.data_analysis.feature_engineering import add_all_features
from cropspectai.modeling.predict import predict_model

# Firmware scripts share imports from their src directory.
FIRMWARE_SRC = config.PROJ_ROOT / "firmware" / "esp32-as7343-spectrometer" / "src"
sys.path.insert(0, str(FIRMWARE_SRC))
import as7343_measure  # noqa: E402
import channel_config  # noqa: E402

# Settings

#model
MODEL_PATH = config.MODELS_DIR / "xgb_v_01.joblib"
MODEL_CONFIG_PATH = config.MODELS_DIR / "xgb_best_parameters_v_01.json"

#trail data
# CSV_FILE = channel_config.MISC_DATA_DIR / "reflectance.csv" # input
# PREDICTION_FILE = config.PREDICTION_DATA_DIR / "misc" / "prediction.csv" # output

#basil experiment data
CSV_FILE = channel_config.RAW_DATA_DIR / "basil_experiment.csv" # input
PREDICTION_FILE = config.BASIL_EXPERIMENT_DATA_PATH / "prediction.csv" # output


CHANNELS = list(config.AS7343_CHANNELS)
PREDICTION_COLUMNS = ["timestamp", "specimen", *CHANNELS, "noisy_channels", "cab"]


def main():
    """Append a prediction for each successfully saved measurement."""
    with MODEL_CONFIG_PATH.open() as handle:
        features = json.load(handle)["features"]
    if not MODEL_PATH.is_file():
        raise FileNotFoundError(MODEL_PATH)

    PREDICTION_FILE.parent.mkdir(parents=True, exist_ok=True)
    new_file = not PREDICTION_FILE.exists() or PREDICTION_FILE.stat().st_size == 0
    if not new_file:
        with PREDICTION_FILE.open(newline="") as handle:
            if next(csv.reader(handle), None) != PREDICTION_COLUMNS:
                raise ValueError(f"Existing CSV header does not match: {PREDICTION_FILE}")

    with PREDICTION_FILE.open("a", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=PREDICTION_COLUMNS)
        if new_file:
            writer.writeheader()
            handle.flush()

        def save_prediction(measurement):
            try:
                X = pd.DataFrame([measurement])[CHANNELS]
                X = add_all_features(X)[features]
                cab = float(predict_model(X, MODEL_PATH)[0])
                if not math.isfinite(cab):
                    raise ValueError("Model returned a non-finite cab value")
            except Exception as exc:
                print(f"  Prediction failed: {exc}. Measurement remains saved in firmware CSV.")
                return

            row = {name: measurement[name] for name in PREDICTION_COLUMNS[:-1]}
            row["cab"] = cab
            writer.writerow(row)
            handle.flush()
            print(f"  cab = {cab:.4f} µg/cm²")

        print(f"Saving predictions to {PREDICTION_FILE}")
        as7343_measure.main(on_measurement=save_prediction, csv_file=CSV_FILE)


if __name__ == "__main__":
    main()
