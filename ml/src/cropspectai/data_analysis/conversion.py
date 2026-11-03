import pandas as pd
from datasets import load_dataset
from pathlib import Path
from cropspectai.ml_config import RAW_DATA_DIR


def convert_hf_csv_to_parquet(dataset_name: str, config_name: str, output_parquet_path: Path, split_name: str | None = None) -> None:
    try:
        # Load the dataset from Hugging Face
        print(f"Loading dataset from Hugging Face: {dataset_name}, split: {split_name}")
        ds = load_dataset(dataset_name, name=config_name, split=split_name)
        
        # Convert the dataset to a pandas DataFrame
        df = ds.to_pandas() # works only when split is specified other than None
        
        # Convert the data to Parquet format
        print(f"Converting Hugging Face data to Parquet and saving to {output_parquet_path}")
        df.to_parquet(output_parquet_path, engine='pyarrow')
        print("Conversion complete.")

    except Exception as e:
        print(f"Error during conversion: {e}")


def convert_local_csv_to_parquet(input_csv_path: Path, output_parquet_path: Path) -> None:
    try:
        # Load the CSV file from local
        print(f"Loading dataset from local: {input_csv_path}")
        df = pd.read_csv(input_csv_path)
        
        # Convert the dataset to a pandas DataFrame
        print(f"Converting local data to Parquet and saving to {output_parquet_path}")
        df.to_parquet(output_parquet_path, engine='pyarrow')
        print("Conversion complete.")

    except Exception as e:
        print(f"Error during conversion: {e}")

def main() -> None:
    
    # Choose between Hugging Face and local data
    use_local_data = True

    if use_local_data:

        # Iterate over all CSV files in the raw folder
        raw_folder_csv = RAW_DATA_DIR / "GreenHyperSpectra" / "csv" / "labeled_splits"
        raw_folder_parquet = RAW_DATA_DIR / "GreenHyperSpectra" / "parquet" / "labeled_splits"

        for csv_file in raw_folder_csv.glob("*.csv"):
            output_parquet_path = raw_folder_parquet / (csv_file.stem + ".parquet")
            convert_local_csv_to_parquet(input_csv_path=csv_file, 
                                        output_parquet_path=output_parquet_path)
    else:
        # Convert data from Hugging Face
        dataset_name = "Avatarr05/GreenHyperSpectra"
        config_name = "unlabeled" # subfolder

        output_parquet_path = (
                PROCESSED_DATA_DIR
                / "GreenHyperSpectra"
                / "labeled_all"
                / "all.parquet"
            )

        convert_hf_csv_to_parquet(dataset_name=dataset_name, 
                                    config_name=config_name, 
                                    output_parquet_path=output_parquet_path, 
                                    split_name=split_name)        

if __name__ == "__main__":
     main()

    