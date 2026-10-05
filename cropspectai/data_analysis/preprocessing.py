import pandas as pd
from cropspectai.config import TARGET_COLUMNS, NON_FEATURE_COLUMNS


def split_dataset_into_X_y(df: pd.DataFrame,) -> tuple[pd.DataFrame, pd.DataFrame]:
    missing_targets = [col for col in TARGET_COLUMNS if col not in df.columns]

    if missing_targets:
        raise ValueError(
            f"DataFrame is missing target columns: {missing_targets}"
        )

    X = df.drop(
        columns=TARGET_COLUMNS + NON_FEATURE_COLUMNS,
        errors="ignore", # drop columns if they exist. If some don't exist, don't raise an error
    )

    y = df[TARGET_COLUMNS]

    return X, y