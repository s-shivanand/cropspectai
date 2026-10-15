"""Measure AS7343 reflectance and predict cab: python scripts/measure_predict.py."""

import csv
import json
import math
import sys

import pandas as pd
from serial.tools import list_ports

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
MODEL_VERSION = "v_01"
MODEL_PATH = config.MODELS_DIR / f"xgb_{MODEL_VERSION}.joblib"
MODEL_CONFIG_PATH = config.MODELS_DIR / f"xgb_best_parameters_{MODEL_VERSION}.json"

#trail data
CSV_FILE = channel_config.MISC_DATA_DIR / "reflectance.csv" # input
PREDICTION_FILE = config.PREDICTION_DATA_DIR / "misc" / f"measure_prediction_{MODEL_VERSION}.csv" # output

#basil experiment data
# CSV_FILE = channel_config.RAW_DATA_DIR / "basil_experiment.csv" # input
# PREDICTION_FILE = config.BASIL_EXPERIMENT_DATA_PATH / f"measure_prediction_{MODEL_VERSION}.csv" # output


CHANNELS = list(config.AS7343_CHANNELS)
MEASUREMENT_COLUMNS = ["timestamp", "specimen", *CHANNELS, "noisy_channels"]


def main():
    """Append a prediction for each successfully saved measurement."""
    if as7343_measure.SERIAL_PORT not in {port.device for port in list_ports.comports()}:
        sys.exit(
            "esp32 device not connected. Connect the ESP32 and check SERIAL_PORT "
            "in as7343_measure.py."
        )

    with MODEL_CONFIG_PATH.open() as handle:
        features = json.load(handle)["features"]
    prediction_columns = list(dict.fromkeys([*MEASUREMENT_COLUMNS, *features, "cab"]))
    if not MODEL_PATH.is_file():
        raise FileNotFoundError(MODEL_PATH)

    PREDICTION_FILE.parent.mkdir(parents=True, exist_ok=True)
    new_file = not PREDICTION_FILE.exists() or PREDICTION_FILE.stat().st_size == 0
    if not new_file:
        with PREDICTION_FILE.open(newline="") as handle:
            existing_columns = next(csv.reader(handle), None)
        if existing_columns == [*MEASUREMENT_COLUMNS, "cab"]:
            existing = pd.read_csv(PREDICTION_FILE, keep_default_na=False)
            X = existing[CHANNELS].apply(pd.to_numeric, errors="raise")
            X = add_all_features(X)[features]
            for feature in features:
                existing[feature] = X[feature]
            existing[prediction_columns].to_csv(PREDICTION_FILE, index=False)
            print("Added model features to existing CSV.")
        elif existing_columns != prediction_columns:
            raise ValueError(
                f"Existing CSV header does not match: {PREDICTION_FILE}. "
                "Choose a new output path or move the old CSV before running again."
            )

    with PREDICTION_FILE.open("a", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=prediction_columns)
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

            row = {name: measurement[name] for name in MEASUREMENT_COLUMNS}
            row.update(X.iloc[0].to_dict())
            row["cab"] = cab
            writer.writerow(row)
            handle.flush()
            print(f"  cab = {cab:.4f} µg/cm²")

        print(f"Saving predictions to {PREDICTION_FILE}")
        as7343_measure.main(on_measurement=save_prediction, csv_file=CSV_FILE)


if __name__ == "__main__":
    main()
