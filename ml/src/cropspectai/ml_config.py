"""Shared paths, dataset columns, and AS7343 metadata for ML workflows.

Paths are derived from this module's repository location rather than the
current working directory. Importing this module also loads environment
variables from an available .env file through python-dotenv.

Attributes:
    PROJ_ROOT: Repository root resolved from this module's location.
    DATA_DIR: Root directory for project datasets.
    RAW_DATA_DIR: Original datasets and raw format conversions.
    DOWNSAMPLED_DATA_DIR: Datasets reduced to sensor channels.
    PROCESSED_DATA_DIR: Cleaned datasets and engineered features.
    EXTERNAL_DATA_DIR: Datasets obtained from external sources.
    PREDICTION_DATA_DIR: Measurement predictions and tracking outputs.
    RAW_PARQUET_DATA_PATH: Raw GreenHyperSpectra Parquet directory.
    RAW_PARQUET_DATA_SPLITS_PATH: Labeled splits of raw Parquet data.
    PROCESSED_PARQUET_DATA_SPLITS_PATH: Labeled splits of processed data.
    DOWNSAMPLED_PARQUET_DATA_SPLITS_PATH: Labeled splits of downsampled data.
    BASIL_EXPERIMENT_DATA_PATH: Basil experiment prediction directory.
    MODELS_DIR: Saved pipelines and model configuration files.
    REPORTS_DIR: Evaluation and analysis reports.
    FIGURES_DIR: Figures generated for reports.
    TARGET_COLUMNS: Supported target column names, in dataset order.
    NON_FEATURE_COLUMNS: Metadata columns excluded from model inputs.
    AS7343_CHANNELS: Ordered channel metadata with peak and FWHM in nanometers.
    channel_colors: Plotting colours keyed by AS7343 channel name.
    FWHM_std_factor: Divisor converting Gaussian FWHM to standard deviation.
"""

from pathlib import Path

from dotenv import load_dotenv

# Load environment variables from .env file if it exists
load_dotenv()

# Paths
PROJ_ROOT: Path = Path(__file__).resolve().parents[3]

DATA_DIR: Path = PROJ_ROOT / "data"
RAW_DATA_DIR: Path = DATA_DIR / "raw"
DOWNSAMPLED_DATA_DIR: Path = DATA_DIR / "downsampled"
PROCESSED_DATA_DIR: Path = DATA_DIR / "processed"
EXTERNAL_DATA_DIR: Path = DATA_DIR / "external"
PREDICTION_DATA_DIR: Path = DATA_DIR / "prediction"

# GHS DATA PATH
RAW_PARQUET_DATA_PATH: Path = RAW_DATA_DIR / "GreenHyperSpectra" / "parquet"
RAW_PARQUET_DATA_SPLITS_PATH: Path = RAW_PARQUET_DATA_PATH / "labeled_splits"
PROCESSED_PARQUET_DATA_SPLITS_PATH: Path = (
    PROCESSED_DATA_DIR / "GreenHyperSpectra" / "parquet" / "labeled_splits"
)
DOWNSAMPLED_PARQUET_DATA_SPLITS_PATH: Path = (
    DOWNSAMPLED_DATA_DIR / "GreenHyperSpectra" / "parquet" / "labeled_splits"
)

BASIL_EXPERIMENT_DATA_PATH: Path = PREDICTION_DATA_DIR / "basil_experiment"

MODELS_DIR: Path = PROJ_ROOT / "models"

REPORTS_DIR: Path = PROJ_ROOT / "reports"
FIGURES_DIR: Path = REPORTS_DIR / "figures"

# DATA - COLUMN INFO

TARGET_COLUMNS: list[str] = [
    "cab",
    "cw",
    "cm",
    "LAI",
    "cp",
    "cbc",
    "car",
    "anth",
]

NON_FEATURE_COLUMNS: list[str] = [
    "Unnamed: 0",
    "dataset",
]


# AS7343 CHANNELS

AS7343_CHANNELS: dict[str, dict[str, int]] = {
    "F1": {"peak": 405, "fwhm": 30},
    "F2": {"peak": 425, "fwhm": 22},
    "FZ": {"peak": 450, "fwhm": 55},
    "F3": {"peak": 475, "fwhm": 30},
    "F4": {"peak": 515, "fwhm": 40},
    "FY": {"peak": 555, "fwhm": 100},
    "F5": {"peak": 550, "fwhm": 35},
    "FXL": {"peak": 600, "fwhm": 80},
    "F6": {"peak": 640, "fwhm": 50},
    "F7": {"peak": 690, "fwhm": 55},
    "F8": {"peak": 745, "fwhm": 60},
    "NIR": {"peak": 855, "fwhm": 54},
}

channel_colors: dict[str, str] = {
    "F1": "purple",
    "F2": "blueviolet",
    "FZ": "blue",
    "F3": "deepskyblue",
    "F4": "limegreen",
    "F5": "green",
    "FY": "yellowgreen",
    "FXL": "orange",
    "F6": "orangered",
    "F7": "red",
    "F8": "darkred",
    "NIR": "grey",
}


# standard conversion factor FWHM to std dev

FWHM_std_factor: float = 2.35482
