"""Load shared AS7343 channel metadata and firmware data paths.

Reads include/channel_config.def at import time without opening a serial port.
Malformed definitions or empty or duplicate channel names raise ValueError.

Attributes:
    CONFIG_FILE: Shared channel definition file also included by main.cpp.
    CHANNEL_NAMES: Sensor channel names in firmware transmission order.
    WAVELENGTHS: Corresponding channel center wavelengths in nanometers.
    BAR_COLORS: Corresponding hexadecimal plotting colours.
    NON_FEATURE_COLUMNS: Measurement metadata retained by CSV processing.
    N_CH: Number of configured channels.
    PROJ_ROOT: Firmware project directory resolved from this module's path.
    DATA_DIR: Root directory for firmware datasets.
    MEASUREMENT_DATA_DIR: Measurement datasets and sample assets.
    REF_DATA_DIR: Dark and white reference CSV directory.
    EXPERIMENT_DATA_DIR: Experiment measurement directory.
    MISC_DATA_DIR: General measurement directory.
    PROCESSED_DATA_DIR: Processed experiment measurement directory.
"""

import re
from pathlib import Path

CONFIG_FILE: Path = Path(__file__).resolve().parent.parent / "include" / "channel_config.def"
CHANNEL_NAMES: list[str] = []
WAVELENGTHS: list[int] = []
BAR_COLORS: list[str] = []
NON_FEATURE_COLUMNS: list[str] = ["timestamp", "specimen", "noisy_channels"]

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
N_CH: int = len(CHANNEL_NAMES)

# Paths
PROJ_ROOT: Path = Path(__file__).resolve().parents[1]

DATA_DIR: Path = PROJ_ROOT / "data"
MEASUREMENT_DATA_DIR: Path = DATA_DIR / "measurement"
REF_DATA_DIR: Path = DATA_DIR / "ref"
EXPERIMENT_DATA_DIR: Path = MEASUREMENT_DATA_DIR / "experiment"
MISC_DATA_DIR: Path = MEASUREMENT_DATA_DIR / "misc"
PROCESSED_DATA_DIR: Path = EXPERIMENT_DATA_DIR / "processed"
