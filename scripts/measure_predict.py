"""Measure AS7343 reflectance and predict cab: python scripts/measure_predict.py."""

import csv
import json
import math
import sys
import time
from contextlib import contextmanager, suppress
from dataclasses import dataclass, field
from datetime import datetime
from threading import Lock

import pandas as pd
from serial.tools import list_ports

from cropspectai import ml_config
from cropspectai.data_analysis.feature_engineering import add_all_features
from cropspectai.modeling.predict import predict_model

# Firmware scripts share imports from their src directory.
FIRMWARE_SRC = ml_config.PROJ_ROOT / "firmware" / "esp32-as7343-spectrometer" / "src"
sys.path.insert(0, str(FIRMWARE_SRC))
import as7343_measure  # noqa: E402
import fimware_config  # noqa: E402

# Settings

#model
MODEL_VERSION = "v_01"
MODEL_PATH = ml_config.MODELS_DIR / f"xgb_{MODEL_VERSION}.joblib"
MODEL_CONFIG_PATH = ml_config.MODELS_DIR / f"xgb_best_parameters_{MODEL_VERSION}.json"

#trial data
CSV_FILE = fimware_config.MISC_DATA_DIR / "reflectance.csv" # input
PREDICTION_FILE = ml_config.PREDICTION_DATA_DIR / "misc" / f"measure_prediction_{MODEL_VERSION}.csv" # output

#basil experiment data
# CSV_FILE = fimware_config.EXPERIMENT_DATA_DIR / "basil_experiment.csv" # input
# PREDICTION_FILE = ml_config.BASIL_EXPERIMENT_DATA_PATH / f"measure_prediction_{MODEL_VERSION}.csv" # output


CHANNELS = list(ml_config.AS7343_CHANNELS)
MEASUREMENT_COLUMNS = ["timestamp", "specimen", *CHANNELS, "noisy_channels"]
_MEASUREMENT_LOCK = Lock()


@dataclass
class MeasurementResult:
    """A saved measurement, its prediction, and user-facing quality warnings."""

    measurement: dict
    cab: float | None = None
    warnings: list[str] = field(default_factory=list)
    prediction_error: str | None = None
    prediction_row: dict | None = None


def _check_device():
    if as7343_measure.SERIAL_PORT not in {port.device for port in list_ports.comports()}:
        raise ConnectionError(
            "Measuring device not connected. Please connect the ESP32 and try again."
        )


@contextmanager
def _prediction_output():
    """Validate the output schema and keep the prediction CSV open for a run."""
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
        yield features, writer, handle


def _save_prediction(measurement, features, writer, handle):
    result = MeasurementResult(measurement=measurement)
    try:
        X = pd.DataFrame([measurement])[CHANNELS]
        X = add_all_features(X)[features]
        cab = float(predict_model(X, MODEL_PATH)[0])
        if not math.isfinite(cab):
            raise ValueError("Model returned a non-finite cab value")
    except Exception as exc:
        result.prediction_error = str(exc)
        print(f"  Prediction failed: {exc}. Measurement remains saved in firmware CSV.")
        return result

    row = {name: measurement[name] for name in MEASUREMENT_COLUMNS}
    row.update(X.iloc[0].to_dict())
    row["cab"] = cab
    writer.writerow(row)
    handle.flush()
    result.cab = cab
    result.prediction_row = row.copy()
    print(f"  cab = {cab:.1f} µg/cm²")
    return result


@contextmanager
def _exclusive_measurement():
    # Firmware helpers use module globals; never let two app sessions overlap.
    if not _MEASUREMENT_LOCK.acquire(blocking=False):
        raise RuntimeError("A measurement is already running. Please wait and try again.")
    try:
        yield
    finally:
        _MEASUREMENT_LOCK.release()


@contextmanager
def _sensor_connection(warnings):
    """Adapt the unchanged firmware helpers to a single, bounded acquisition."""
    sensor = as7343_measure
    missing = object()
    previous = {name: getattr(sensor, name, missing) for name in ("DARK_REF", "WHITE_REF", "ser")}
    connection = None
    try:
        sensor.DARK_REF, sensor.WHITE_REF = sensor.load_references(sensor.REF_FILE)
        invalid = [
            name for name, dark, white in zip(
                sensor.CHANNEL_NAMES, sensor.DARK_REF, sensor.WHITE_REF
            ) if white <= dark
        ]
        if invalid:
            warnings.append("White reference <= dark reference; reflectance set to 0 for: "
                            + ", ".join(invalid))
        connection = sensor.serial.Serial(
            sensor.SERIAL_PORT, sensor.BAUD_RATE, timeout=1, write_timeout=1
        )
        sensor.ser = connection
        # Match the boot wait and streaming setup used by the terminal workflow.
        time.sleep(2)
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline:
            if connection.readline().decode("utf-8", errors="ignore").strip() == "AS7343_READY":
                break
        else:
            warnings.append("ESP32 readiness was not confirmed. If acquisition fails, "
                            "press RST on the device and try again.")
        connection.write(b"s")
        time.sleep(0.3)
        connection.reset_input_buffer()
        yield
    finally:
        try:
            if connection is not None:
                try:
                    with suppress(sensor.serial.SerialException, OSError):
                        connection.write(b"s")
                finally:
                    connection.close()
        finally:
            for name, value in previous.items():
                if value is missing:
                    sensor.__dict__.pop(name, None)
                else:
                    setattr(sensor, name, value)


def measure_and_predict(specimen: str) -> MeasurementResult:
    """Measure one specimen and append to the same CSVs as the terminal script.

    Acquires the existing ten-reading batch and closes the port after each call.
    A prediction failure returns ``cab=None`` and ``prediction_error``; the
    measurement remains saved. Acquisition and file errors raise exceptions.
    """
    specimen = specimen.strip()
    if not specimen:
        raise ValueError("Please enter a specimen label before measuring.")

    with _exclusive_measurement():
        _check_device()
        warnings = []
        with _prediction_output() as (features, prediction_writer, prediction_handle):
            sensor = as7343_measure
            CSV_FILE.parent.mkdir(parents=True, exist_ok=True)
            new_file = not CSV_FILE.exists() or CSV_FILE.stat().st_size == 0
            if not new_file:
                with CSV_FILE.open(newline="") as handle:
                    if next(csv.reader(handle), None) != sensor.HEADER:
                        raise ValueError(f"Existing CSV header does not match: {CSV_FILE}")

            with _sensor_connection(warnings):
                raw = sensor.take_measurement()
                if raw is None:
                    raise RuntimeError("Measurement failed or timed out. Nothing was measured "
                                       "successfully or saved. Check the sensor and try again.")
                reflectance, stds = sensor.normalized_statistics(raw)
                noisy = sensor.check_noise(reflectance, stds)
                if noisy:
                    warnings.append("Noisy channels: " + "; ".join(noisy))
                if max(value for reading in raw for value in reading) > 60000:
                    warnings.append("A channel is near saturation. Lower the gain or LED current.")
                row = (
                    [datetime.now().isoformat(timespec="seconds"), specimen]
                    + [round(value, 6) for value in reflectance]
                    + ["; ".join(noisy)]
                    + list(sensor.SENSOR_SETTINGS.values())
                )
                measurement = dict(zip(sensor.HEADER, row))
                with CSV_FILE.open("a", newline="") as handle:
                    writer = csv.writer(handle)
                    if new_file:
                        writer.writerow(sensor.HEADER)
                    writer.writerow(row)

            result = _save_prediction(measurement, features, prediction_writer, prediction_handle)
            result.warnings = warnings
            return result


def main():
    """Keep the existing interactive label/Enter/q measurement workflow."""
    try:
        _check_device()
    except ConnectionError as exc:
        sys.exit(str(exc))

    with _exclusive_measurement(), _prediction_output() as (features, writer, handle):
        def save_prediction(measurement):
            return _save_prediction(measurement, features, writer, handle)

        print(f"Saving predictions to {PREDICTION_FILE}")
        as7343_measure.main(on_measurement=save_prediction, csv_file=CSV_FILE)


if __name__ == "__main__":
    main()
