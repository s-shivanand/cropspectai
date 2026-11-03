from pathlib import Path

from dotenv import load_dotenv

# Load environment variables from .env file if it exists
load_dotenv()

# Paths
PROJ_ROOT = Path(__file__).resolve().parents[3]

DATA_DIR = PROJ_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
DOWNSAMPLED_DATA_DIR = DATA_DIR / "downsampled"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
EXTERNAL_DATA_DIR = DATA_DIR / "external"
PREDICTION_DATA_DIR = DATA_DIR / "prediction"

# GHS DATA PATH
RAW_PARQUET_DATA_SPLITS_PATH = RAW_DATA_DIR / "GreenHyperSpectra" / "parquet" / "labeled_splits"
PROCESSED_PARQUET_DATA_SPLITS_PATH = PROCESSED_DATA_DIR / "GreenHyperSpectra" / "parquet" / "labeled_splits"
DOWNSAMPLED_PARQUET_DATA_SPLITS_PATH = DOWNSAMPLED_DATA_DIR / "GreenHyperSpectra" / "parquet" / "labeled_splits"

BASIL_EXPERIMENT_DATA_PATH = PREDICTION_DATA_DIR / "basil_experiment"

MODELS_DIR = PROJ_ROOT / "models"

REPORTS_DIR = PROJ_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"

# DATA - COLUMN INFO

TARGET_COLUMNS = [
    "cab",
    "cw",
    "cm",
    "LAI",
    "cp",
    "cbc",
    "car",
    "anth",
]

NON_FEATURE_COLUMNS = [
    "Unnamed: 0",
    "dataset",
]


# AS7343 CHANNELS

AS7343_CHANNELS = {
    "F1":  {"peak": 405, "fwhm": 30},
    "F2":  {"peak": 425, "fwhm": 22},
    "FZ":  {"peak": 450, "fwhm": 55},
    "F3":  {"peak": 475, "fwhm": 30},
    "F4":  {"peak": 515, "fwhm": 40},
    "FY":  {"peak": 555, "fwhm": 100},
    "F5":  {"peak": 550, "fwhm": 35},
    "FXL": {"peak": 600, "fwhm": 80},
    "F6":  {"peak": 640, "fwhm": 50},
    "F7":  {"peak": 690, "fwhm": 55},
    "F8":  {"peak": 745, "fwhm": 60},
    "NIR": {"peak": 855, "fwhm": 54},
}

channel_colors = {
        "F1":  "purple",
        "F2":  "blueviolet",
        "FZ":  "blue",
        "F3":  "deepskyblue",
        "F4":  "limegreen",
        "F5":  "green",
        "FY":  "yellowgreen",
        "FXL": "orange",
        "F6":  "orangered",
        "F7":  "red",
        "F8":  "darkred",
        "NIR": "grey",
    }


# standard conversion factor FWHM to std dev

FWHM_std_factor = 2.35482 
