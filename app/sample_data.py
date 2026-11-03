"""Predict saved example spectra without connecting to or saving sensor data."""

import json
import math

import pandas as pd

from scripts import measure_predict as workflow


SAMPLES_DIR = workflow.fimware_config.MEASUREMENT_DATA_DIR / "samples"
SAMPLE_FILE = SAMPLES_DIR / "sample_measurements.csv"


def load_samples():
    return pd.read_csv(SAMPLE_FILE, keep_default_na=False)


def predict_sample(sample, specimen):
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
