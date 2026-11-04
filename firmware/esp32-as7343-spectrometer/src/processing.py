"""Select measurement metadata and channel columns from saved sensor CSVs."""

from pathlib import Path

import pandas as pd

import fimware_config
from fimware_config import CHANNEL_NAMES, NON_FEATURE_COLUMNS


def split_dataset_into_X(path: str | Path) -> pd.DataFrame:
    """Load a measurement CSV and select metadata and spectral channels.

    Args:
        path: Source measurement CSV file.

    Returns:
        DataFrame containing NON_FEATURE_COLUMNS followed by CHANNEL_NAMES,
        in that order. Despite the function name, measurement metadata is
        retained; sensor settings and other columns are excluded.

    Raises:
        FileNotFoundError: The source CSV is missing.
        KeyError: A configured metadata or channel column is missing.

    Note:
        Prints the shape of the imported dataset. The source file is unchanged.
    """
    df = pd.read_csv(path)
    print("Shape of imported data: ", df.shape)
    column_list = NON_FEATURE_COLUMNS + CHANNEL_NAMES
    return df[column_list].copy()


def save_processed_data(X: pd.DataFrame, path: str | Path) -> None:
    """Write selected measurement columns to a CSV without the row index.

    Args:
        X: DataFrame to save, including any retained measurement metadata.
        path: Destination CSV file. Its parent directory must already exist.

    Raises:
        OSError: The destination cannot be written.

    Note:
        An existing destination file is overwritten.
    """
    X.to_csv(path, index=False)


if __name__ == "__main__":
    # basil data
    # RAW_FILE = fimware_config.EXPERIMENT_DATA_DIR /"basil_experiment.csv"
    # output_path = fimware_config.PROCESSED_DATA_DIR / "X_new.csv"

    # misc data
    RAW_FILE = fimware_config.MISC_DATA_DIR / "reflectance.csv"
    output_path = fimware_config.MISC_DATA_DIR / "X_new.csv"

    X = split_dataset_into_X(RAW_FILE)
    print(f"Shape after splitting: {X.shape}")
    print(X.head())
    save_processed_data(X, output_path)
