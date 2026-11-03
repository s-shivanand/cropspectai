import fimware_config
from fimware_config import CHANNEL_NAMES, NON_FEATURE_COLUMNS
import pandas as pd


def split_dataset_into_X(path):
    df = pd.read_csv(path)
    print("Shape of imported data: ", df.shape)
    column_list = NON_FEATURE_COLUMNS + CHANNEL_NAMES
    return df[column_list].copy()

def save_processed_data(X, path):
    X.to_csv(path, index=False)


if __name__ == "__main__":
    # basil data
    # RAW_FILE = fimware_config.RAW_DATA_DIR /"basil_experiment.csv"
    # output_path = fimware_config.PROCESSED_DATA_DIR / "X_new.csv"

    #misc data
    RAW_FILE = fimware_config.MISC_DATA_DIR /"reflectance.csv"
    output_path = fimware_config.MISC_DATA_DIR / "X_new.csv"
    
    X = split_dataset_into_X(RAW_FILE)
    print(f"Shape after splitting: {X.shape}")
    print(X.head())
    save_processed_data(X, output_path)

