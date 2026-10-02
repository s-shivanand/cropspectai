import pandas as pd
from cropspectai.config import TARGET_COLUMNS, NON_FEATURE_COLUMNS


def split_dataset_into_X_y(df: pd.DataFrame,) -> tuple[pd.DataFrame, pd.DataFrame]:
    X = df.drop(columns=TARGET_COLUMNS + NON_FEATURE_COLUMNS)
    y = df[TARGET_COLUMNS]

    return X, y