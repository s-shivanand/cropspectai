"""Evaluate saved regression pipelines on held-out labeled data."""

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    mean_absolute_error,
    mean_absolute_percentage_error,
    r2_score,
    root_mean_squared_error,
)


def evaluate_model(
    X: pd.DataFrame | np.ndarray,
    y: pd.DataFrame | pd.Series | np.ndarray,
    model_path: str | Path,
) -> dict[str, float]:
    """Calculate test-set metrics using a saved regression pipeline.

    Args:
        X: Prepared feature matrix, with columns in the saved model's order.
        y: Target values in the same sample order as ``X``. Single-target
            inputs may be one-dimensional or a single-column DataFrame.
        model_path: Path to the pipeline serialized with Joblib.

    Returns:
        Dictionary containing test MAE, MAPE in percent, RMSE, and R-squared.
        Keys retain the human-readable labels used in reports. MAE and RMSE
        are expressed in the target's units.

    Raises:
        FileNotFoundError: The saved model file does not exist.
        ValueError: Features or targets are incompatible with the pipeline
            or metric calculations.

    Note:
        The saved pipeline is loaded and used without refitting. The input
        data is not modified, and no results are saved to disk.
    """
    # load best model
    pipeline = joblib.load(model_path)

    # Predict
    predictions_test = pipeline.predict(X)

    # Test metrics
    mae_test = mean_absolute_error(y, predictions_test)
    mape_test = mean_absolute_percentage_error(y, predictions_test) * 100
    r2_test = r2_score(y, predictions_test)
    rmse_test = root_mean_squared_error(y, predictions_test)

    metrics = {
        "MAE (test)": mae_test,
        "MAPE (test) [%]": mape_test,
        "RMSE (test)": rmse_test,
        "R² (test)": r2_test,
    }

    return metrics
