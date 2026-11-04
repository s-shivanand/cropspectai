"""Print dataset summaries and checks for missing or out-of-range reflectance."""

import pandas as pd

from cropspectai.data_analysis.preprocessing import split_dataset_into_X_y
from cropspectai.ml_config import RAW_PARQUET_DATA_SPLITS_PATH


def explore_dataset(df: pd.DataFrame) -> None:
    """Print a structural and statistical summary of a dataset.

    Args:
        df: DataFrame to summarize, including any feature and target columns.

    Note:
        Prints shape, dtypes, missing-value counts and percentages, duplicate
        statistics, and the pandas descriptive summary. The input is unchanged.
    """
    print("=" * 50)
    print("Shape")
    print("=" * 50)
    print(f"Rows: {df.shape[0]:,}")
    print(f"Columns: {df.shape[1]}")

    print("=" * 50)
    print("Data Types")
    print("=" * 50)
    print(df.dtypes)

    print("=" * 50)
    print("Missing Values")
    print("=" * 50)
    null_counts = df.isnull().sum()
    null_pct = (df.isnull().sum() / len(df) * 100).round(1)
    null_df = pd.DataFrame({"Missing": null_counts, "Percent": null_pct})
    print(null_df[null_df["Missing"] > 0].sort_values("Missing", ascending=False))

    print("=" * 50)
    print("Duplicate Rows")
    print("=" * 50)
    duplicate_mask = df.duplicated()
    duplicate_rows = duplicate_mask.sum()
    duplicate_pct = (duplicate_rows / len(df) * 100).round(1)
    duplicates = df.loc[duplicate_mask]
    duplicates_all_nan = duplicates.isna().all(axis=1).sum()  # NaN checks among duplicated rows
    print(f"Duplicates:              {duplicate_rows}")
    print(f"Percent:                 {duplicate_pct}%")
    print(f"Duplicates all NaN:      {duplicates_all_nan}")

    print("=" * 50)
    print("Statistical Summary")
    print("=" * 50)
    print(df.describe())


def validate_spectral_values(df: pd.DataFrame) -> None:
    """Print missing-value and spectral-range checks for a labeled dataset.

    Args:
        df: DataFrame containing spectral features and all configured targets.

    Raises:
        ValueError: Any configured target is absent.

    Note:
        Reports missing values in features and targets, then checks whether
        feature values fall outside [0, 1]. Values are not changed. This is a
        reporting helper: range violations do not raise exceptions, and missing
        values alone do not prevent the within-range message.
    """
    X, y = split_dataset_into_X_y(df)

    # Check missing values in both X and y
    for name, data in [("Spectral features (X)", X), ("Target (y)", y)]:
        if data.isna().any().any():
            print(f"\n{name} contains missing values.")

    # Check value range only in X
    if (X > 1).any().any():
        print("\nSpectral features (X) contains values greater than 1.")

    if (X < 0).any().any():
        print("\nSpectral features (X) contains negative values.")

    if not (X > 1).any().any() and not (X < 0).any().any():
        print("\nSpectral features (X): all values are between 0 and 1.")


if __name__ == "__main__":
    train_path = RAW_PARQUET_DATA_SPLITS_PATH / "train.parquet"
    test_path = RAW_PARQUET_DATA_SPLITS_PATH / "test.parquet"

    # train
    print("Train dataset:")
    df_train = pd.read_parquet(train_path)
    validate_spectral_values(df_train)
    explore_dataset(df_train)

    # test
    print("Test dataset:")
    df_test = pd.read_parquet(test_path)
    validate_spectral_values(df_test)
    explore_dataset(df_test)
