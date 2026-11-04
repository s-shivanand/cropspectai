"""Load saved regression pipelines and predict from prepared features."""

from pathlib import Path

import joblib
import numpy as np
import pandas as pd


def predict_model(X: pd.DataFrame | np.ndarray, model_path: str | Path) -> np.ndarray:
    """Predict target values with a saved model pipeline.

    Args:
        X: Numeric feature matrix with samples as rows. Feature names and
            order must match those used to train the saved pipeline.
        model_path: Path to the pipeline serialized with Joblib.

    Returns:
        Prediction array with shape (n_samples,) for a single target, or
        (n_samples, n_targets) for a pipeline supporting multiple targets.

    Raises:
        FileNotFoundError: The saved model file does not exist.
        ValueError: The pipeline rejects the supplied feature matrix.

    Note:
        Loads the pipeline on every call. Any fitted preprocessing inside
        that pipeline is applied during prediction; feature engineering
        must already have been performed by the caller.
    """
    # load best model
    pipeline = joblib.load(model_path)

    # Predict
    predictions = pipeline.predict(X)

    return predictions
