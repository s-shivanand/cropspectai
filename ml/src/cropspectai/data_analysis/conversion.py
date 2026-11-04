"""Convert local or Hugging Face CSV datasets to Parquet files."""

from pathlib import Path

from datasets import load_dataset
import pandas as pd

from cropspectai.ml_config import RAW_DATA_DIR, RAW_PARQUET_DATA_PATH, RAW_PARQUET_DATA_SPLITS_PATH


def convert_hf_csv_to_parquet(
    dataset_name: str,
    config_name: str,
    output_parquet_path: str | Path,
    split_name: str | None = None,
) -> None:
    """Convert one Hugging Face dataset split to a Parquet file.

    Args:
        dataset_name: Dataset identifier on the Hugging Face Hub.
        config_name: Dataset configuration or subset name.
        output_parquet_path: Destination file. Its parent directory must exist.
        split_name: Split to load, such as "train". A single split is needed
            because the loaded object must provide ``to_pandas()``.

    Note:
        Loading or conversion errors are printed and caught rather than raised.
        Omitting the split can return a dataset dictionary that this helper
        cannot convert directly. The Parquet writer requires the existing
        PyArrow dependency.
    """
    try:
        # Load the dataset from Hugging Face
        print(f"Loading dataset from Hugging Face: {dataset_name}, split: {split_name}")
        ds = load_dataset(dataset_name, name=config_name, split=split_name)

        # Convert the dataset to a pandas DataFrame
        df = ds.to_pandas()  # works only when split is specified other than None

        # Convert the data to Parquet format
        print(f"Converting Hugging Face data to Parquet and saving to {output_parquet_path}")
        df.to_parquet(output_parquet_path, engine="pyarrow")
        print("Conversion complete.")

    except Exception as e:
        print(f"Error during conversion: {e}")


def convert_local_csv_to_parquet(
    input_csv_path: str | Path, output_parquet_path: str | Path
) -> None:
    """Convert a local CSV file to Parquet using PyArrow.

    Args:
        input_csv_path: Source CSV file.
        output_parquet_path: Destination file. Its parent directory must exist.

    Note:
        Read or write errors are printed and caught rather than raised. An
        existing destination file is overwritten on a successful conversion.
    """
    try:
        # Load the CSV file from local
        print(f"Loading dataset from local: {input_csv_path}")
        df = pd.read_csv(input_csv_path)

        # Convert the dataset to a pandas DataFrame
        print(f"Converting local data to Parquet and saving to {output_parquet_path}")
        df.to_parquet(output_parquet_path, engine="pyarrow")
        print("Conversion complete.")

    except Exception as e:
        print(f"Error during conversion: {e}")


def main() -> None:
    """Run CSV-to-Parquet conversion using the settings in this module.

    The default branch converts CSV files in the configured raw dataset folder.
    Destination directories must already exist. The optional Hugging Face
    branch writes its configured split to the raw Parquet directory.
    """
    # Choose between Hugging Face and local data
    use_local_data = True

    if use_local_data:
        # Iterate over all CSV files in the raw folder
        raw_folder_csv = RAW_DATA_DIR / "GreenHyperSpectra" / "csv" / "labeled_splits"
        raw_folder_parquet = RAW_PARQUET_DATA_SPLITS_PATH

        for csv_file in raw_folder_csv.glob("*.csv"):
            output_parquet_path = raw_folder_parquet / (csv_file.stem + ".parquet")
            convert_local_csv_to_parquet(
                input_csv_path=csv_file, output_parquet_path=output_parquet_path
            )
    else:
        # Convert data from Hugging Face
        dataset_name = "Avatarr05/GreenHyperSpectra"
        config_name = "unlabeled"  # subfolder
        split_name = "train"

        output_parquet_path = RAW_PARQUET_DATA_PATH / "all.parquet"

        convert_hf_csv_to_parquet(
            dataset_name=dataset_name,
            config_name=config_name,
            output_parquet_path=output_parquet_path,
            split_name=split_name,
        )


if __name__ == "__main__":
    main()
