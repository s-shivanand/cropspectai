"""Load shared sensor channel metadata without opening a serial connection."""
import re
from pathlib import Path

CONFIG_FILE = Path(__file__).resolve().parent.parent / "include" / "channel_config.def"
CHANNEL_NAMES = []
WAVELENGTHS = []
BAR_COLORS = []
NON_FEATURE_COLUMNS = ["timestamp", "specimen", "noisy_channels"]

for line in CONFIG_FILE.read_text().splitlines():
    line = line.strip()
    if not line or line.startswith("//"):
        continue
    match = re.fullmatch(r'CHANNEL\((\w+),\s*(\d+),\s*"(#[0-9A-Fa-f]{6})"\)', line)
    if match is None:
        raise ValueError(f"Invalid channel configuration: {line}")
    name, wavelength, color = match.groups()
    CHANNEL_NAMES.append(name)
    WAVELENGTHS.append(int(wavelength))
    BAR_COLORS.append(color)

if not CHANNEL_NAMES or len(set(CHANNEL_NAMES)) != len(CHANNEL_NAMES):
    raise ValueError("Channel configuration must contain unique channel names")
N_CH = len(CHANNEL_NAMES)

# Paths
PROJ_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = PROJ_ROOT / "data"
MEASUREMENT_DATA_DIR = DATA_DIR / "measurement"
REF_DATA_DIR = DATA_DIR / "ref"
EXPERIMENT_DATA_DIR = MEASUREMENT_DATA_DIR / "experiment"
MISC_DATA_DIR = MEASUREMENT_DATA_DIR / "misc"
PROCESSED_DATA_DIR = EXPERIMENT_DATA_DIR / "processed"

