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


def drop_duplicates(df: pd.DataFrame, target: str=None) -> pd.DataFrame:
    print(f"(Before) Shape of dataframe: {df.shape}")
    print("Dropping duplicate rows")
    duplicate_count = df.duplicated().sum()
    print(f"Duplicate rows: {duplicate_count}")

    df = df.drop_duplicates().copy()
    print(f"(After) Shape of dataframe: {df.shape}")

    print("Checking for rows without labels")

    _, y = split_dataset_into_X_y(df)

    if target is None:

        # Rows where all target values are NaN
        no_label_mask = y.isna().all(axis=1)

    else:
        no_label_mask = y[target].isna()

    print(f"Rows with no labels: {no_label_mask.sum()}")
     
    # Remove rows with no labels
    df = df.loc[~no_label_mask].copy()   
    print(f"(Before) Shape of dataframe: {df.shape}")   


    return df