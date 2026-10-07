import pandas as pd
from cropspectai.config import TARGET_COLUMNS, NON_FEATURE_COLUMNS


def split_dataset_into_X_y(df: pd.DataFrame, target: str | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:

    if target is not None:
        # A specific target was requested
        if target not in TARGET_COLUMNS:
            raise ValueError(
                f"'{target}' is not a valid target. "
                f"Expected one of: {TARGET_COLUMNS}"
            )

        if target not in df.columns:
            raise ValueError(
                f"Target column '{target}' not found in DataFrame."
            )

        target_columns = [target]

    else:
        # No specific target requested, so all targets must be present
        missing_targets = [
            col for col in TARGET_COLUMNS
            if col not in df.columns
        ]

        if missing_targets:
            raise ValueError(
                f"DataFrame is missing target columns: {missing_targets}"
            )

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

    print(f"Rows with no labels: {no_label_mask.sum()}")

    if target is not None:
        # Reconstruct dataframe with features + requested target only.
        # Other target columns are thereby removed.
        df = pd.concat([X, y], axis=1)

    df = df.loc[~no_label_mask].copy()

    print(f"(After) Shape of dataframe: {df.shape}")

    return df