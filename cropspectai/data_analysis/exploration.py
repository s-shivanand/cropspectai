import pandas as pd

from cropspectai.config import RAW_PARQUET_DATA_SPLITS_PATH
from cropspectai.data_analysis.preprocessing import split_dataset_into_X_y


def explore_dataset(df: pd.DataFrame) -> None:
    print(f"Shape: {df.shape}\n")

    print("Data types:")
    print(df.dtypes)

    print("\nNull values:")
    print(df.isna().sum()) # isna() same as isnull() 

    print("\nDuplicate rows:")
    print(df.duplicated().sum())


def validate_spectral_values(features: pd.DataFrame) -> None:
    if features.isna().any().any():
        raise ValueError("\nSpectral features contain missing values.")

    if (features > 1).any().any():
        raise ValueError("\nSpectral features contain values greater than 1.")

    if (features < 0).any().any():
        raise ValueError("\nSpectral features contain negative values.")

    print("\nAll spectral values are between 0 and 1.")
    print("\nSpectral features contain no missing values.")


def load_and_explore_datasets(
    train_path,
    test_path,
) -> None:
    df_train = pd.read_parquet(train_path)
    df_test = pd.read_parquet(test_path)

    print("=== TRAIN DATASET ===")
    explore_dataset(df_train)

    X_train, _ = split_dataset_into_X_y(df_train)
    validate_spectral_values(X_train)

    print("\n=== TEST DATASET ===")
    explore_dataset(df_test)

    X_test, _ = split_dataset_into_X_y(df_test)
    validate_spectral_values(X_test)


if __name__ == "__main__":
    train_path = RAW_PARQUET_DATA_SPLITS_PATH / "train.parquet"
    test_path = RAW_PARQUET_DATA_SPLITS_PATH / "test.parquet"

    load_and_explore_datasets(train_path, test_path)