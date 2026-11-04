"""Predict saved example spectra without connecting to or saving sensor data."""

import json
import math
from collections.abc import Mapping
from pathlib import Path

import pandas as pd

from scripts import measure_predict as workflow

SAMPLES_DIR: Path = workflow.fimware_config.MEASUREMENT_DATA_DIR / "samples"
SAMPLE_FILE: Path = SAMPLES_DIR / "sample_measurements.csv"


def load_samples() -> pd.DataFrame:
    """Read the bundled sample measurements for the device-free workflow.

    Returns:
        DataFrame containing timestamps, specimen labels, and spectral channels.
        Empty fields remain empty strings rather than being converted to NaN.

    Raises:
        FileNotFoundError: The configured sample CSV does not exist.
    """
    return pd.read_csv(SAMPLE_FILE, keep_default_na=False)


def predict_sample(sample: Mapping[str, object], specimen: str) -> workflow.MeasurementResult:
    """Predict Cab for one sample using the configured model and features.

    Args:
        sample: Measurement mapping containing timestamp, specimen, and all
            configured reflectance channels.
        specimen: Label to use in the returned measurement; whitespace is trimmed.

    Returns:
        MeasurementResult containing the copied measurement and finite Cab
        prediction. No measurement, prediction, or tracking file is written.

    Raises:
        FileNotFoundError: The model or its feature configuration is missing.
        KeyError: A required sample channel or configured feature is absent.
        ValueError: Prediction fails validation or produces a nonfinite Cab value.
    """
    measurement = dict(sample)
    measurement["specimen"] = specimen.strip()
    with workflow.MODEL_CONFIG_PATH.open() as handle:
        features = json.load(handle)["features"]
    X = pd.DataFrame([measurement])[workflow.CHANNELS]
    X = workflow.add_all_features(X)[features]
    cab = float(workflow.predict_model(X, workflow.MODEL_PATH)[0])
    if not math.isfinite(cab):
        raise ValueError("Model returned a non-finite cab value")
    return workflow.MeasurementResult(measurement=measurement, cab=cab)
