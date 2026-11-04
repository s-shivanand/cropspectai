"""Separate labels, remove duplicate or unlabeled rows, and cap target values."""

from collections.abc import Sequence

import pandas as pd

from cropspectai.ml_config import NON_FEATURE_COLUMNS, TARGET_COLUMNS


def split_dataset_into_X_y(
    df: pd.DataFrame, target: str | None = None
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Separate spectral features from the requested target columns.

    Args:
        df: Dataset containing spectral features and configured target columns.
        target: Single target to return, or None to return all configured targets.

    Returns:
        Pair of DataFrames ``(X, y)`` retaining the input index. ``X`` excludes
        all configured target and non-feature columns that are present. ``y``
        contains the selected target or all targets in configuration order.

    Raises:
        ValueError: The requested target is unknown or absent, or any configured
            target is missing when ``target`` is None.
    """
    if target is not None:
        # A specific target was requested
        if target not in TARGET_COLUMNS:
            raise ValueError(
                f"'{target}' is not a valid target. Expected one of: {TARGET_COLUMNS}"
            )

        if target not in df.columns:
            raise ValueError(f"Target column '{target}' not found in DataFrame.")

        target_columns = [target]

    else:
        # No specific target requested, so all targets must be present
        missing_targets = [col for col in TARGET_COLUMNS if col not in df.columns]

        if missing_targets:
            raise ValueError(f"DataFrame is missing target columns: {missing_targets}")

        target_columns = TARGET_COLUMNS

    # Remove ALL target columns that are present from X
    X = df.drop(
        columns=TARGET_COLUMNS + NON_FEATURE_COLUMNS,
        errors="ignore",
    )

    # Keep only requested target(s) in y
    y = df[target_columns].copy()

    return X, y


def drop_duplicates(df: pd.DataFrame, target: str | None = None) -> pd.DataFrame:
    """Remove duplicate rows and rows without the requested labels.

    Args:
        df: Dataset containing features and target columns.
        target: Single target to require, or None to consider all targets.

    Returns:
        DataFrame with exact duplicate rows removed and the original index
        retained. With one target, only features and that target are retained,
        and rows missing that label are removed. With all targets, a row is
        removed only when every target is missing; other columns are retained.

    Raises:
        ValueError: Target selection fails in ``split_dataset_into_X_y``.

    Note:
        Prints row counts and shapes. The input DataFrame is not modified.
    """
    print(f"(Before) Shape of dataframe: {df.shape}")
    print("Dropping duplicate rows")

    duplicate_count = df.duplicated().sum()
    print(f"Duplicate rows: {duplicate_count}")

    df = df.drop_duplicates().copy()

    print(f"(After) Shape of dataframe: {df.shape}")
    print("Checking for rows without labels")

    X, y = split_dataset_into_X_y(df, target)

    # For all targets:
    # remove row only if ALL targets are missing.
    #
    # For one target:
    # remove row if that target is missing.
    no_label_mask = y.isna().all(axis=1)

    print(f"Rows with no labels: {no_label_mask.sum()} with target: {target}")

    if target is not None:
        # Reconstruct dataframe with features + requested target only.
        # Other target columns are thereby removed.
        df = pd.concat([X, y], axis=1)

    df = df.loc[~no_label_mask].copy()

    print(f"(After) Shape of dataframe: {df.shape}")

    return df


def find_sample_count_above_threshold(
    df: pd.DataFrame, target: str, thresholds: Sequence[float]
) -> None:
    """Print the number of target values above each threshold.

    Args:
        df: DataFrame containing the target column.
        target: Name of the column to count.
        thresholds: Numeric upper bounds to test using strict greater-than
            comparisons. Missing target values do not contribute to the counts.

    Raises:
        KeyError: The target column is absent.

    Note:
        The current printed label is "cab" regardless of the selected target.
        Counts are printed rather than returned.
    """
    col = df[target]

    for threshold in thresholds:
        count = (col > threshold).sum()
        print(f"cab > {threshold}: {count} samples")


def clip_values_above_threshold(df: pd.DataFrame, target: str, threshold: float) -> pd.Series:
    """Return a copy of one target column with values capped at a threshold.

    Args:
        df: DataFrame containing the target column.
        target: Name of the column to clip.
        threshold: Numeric upper bound. Values at or below it are retained.

    Returns:
        Clipped target Series retaining its name, index, and missing values.
        The input DataFrame is not modified.

    Raises:
        KeyError: The target column is absent.
    """
    return (
        df[target].clip(upper=threshold).copy()
    )  # values greater than threshold are replaced with threshold
