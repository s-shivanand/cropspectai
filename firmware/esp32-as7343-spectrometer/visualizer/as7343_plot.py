"""Display live AS7343 reflectance bars using saved calibration references.

Set SERIAL_PORT and REF_FILE before running. This script loads references,
opens the device, and starts Matplotlib at module level, including on import.
It sends "l" to start streaming and "s" when the plot session closes.
"""

import csv
import math
import sys
from collections.abc import Sequence
from pathlib import Path

import matplotlib.pyplot as plt
import serial
from matplotlib.animation import FuncAnimation
from matplotlib.container import BarContainer

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from fimware_config import BAR_COLORS, CHANNEL_NAMES, N_CH, WAVELENGTHS  # noqa: E402

# ============================================================
# SETTINGS
# ============================================================

SERIAL_PORT: str = "/dev/cu.usbserial-0001"  # CHANGE THIS
BAUD_RATE: int = 115200
REF_FILE: Path = Path(__file__).resolve().parent.parent / "data" / "ref" / "REF.csv"


def load_references(ref_file: str | Path) -> tuple[list[float], list[float]]:
    """Load dark and white raw-count references in configured channel order.

    Args:
        ref_file: CSV containing specimen labels and all configured channels.
            BLACK_REF and DARK_REF are accepted as the dark reference label.

    Returns:
        Pair of dark and white reference lists, each ordered by CHANNEL_NAMES.

    Raises:
        FileNotFoundError: The reference file does not exist.
        ValueError: Required columns or references are missing, reference labels
            are duplicated, or channel values are nonnumeric or nonfinite.
    """
    references = {}
    with open(ref_file, newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"specimen", *CHANNEL_NAMES}
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"Reference CSV is missing columns: {sorted(missing)}")
        for row in reader:
            label = row["specimen"].strip().upper()
            if label not in {"BLACK_REF", "DARK_REF", "WHITE_REF"}:
                continue
            key = "WHITE_REF" if label == "WHITE_REF" else "DARK_REF"
            if key in references:
                raise ValueError(f"Reference CSV contains multiple {key} rows")
            values = [float(row[name]) for name in CHANNEL_NAMES]
            if not all(math.isfinite(value) for value in values):
                raise ValueError(f"Reference CSV contains non-finite values for {label}")
            references[key] = values
    if set(references) != {"DARK_REF", "WHITE_REF"}:
        raise ValueError("Reference CSV needs BLACK_REF (or DARK_REF) and WHITE_REF rows")
    return references["DARK_REF"], references["WHITE_REF"]


def normalize(raw_values: Sequence[float]) -> list[float]:
    """Convert one raw reading to reflectance using global references.

    Args:
        raw_values: Raw channel counts in CHANNEL_NAMES order.

    Returns:
        Reflectance values clamped to [0, 1]. Channels with invalid reference
        ranges produce zero.

    Raises:
        ValueError: The reading does not contain N_CH values.
    """
    if len(raw_values) != N_CH:
        raise ValueError(f"Expected {N_CH} channel values")
    return [
        max(0.0, min(1.0, (raw - dark) / (white - dark))) if white > dark else 0.0
        for raw, dark, white in zip(raw_values, DARK_REF, WHITE_REF)
    ]


DARK_REF, WHITE_REF = load_references(REF_FILE)
print(f"Loaded dark and white references from {REF_FILE}")
invalid_channels = [
    name for name, dark, white in zip(CHANNEL_NAMES, DARK_REF, WHITE_REF) if white <= dark
]
if invalid_channels:
    print("Warning: white <= dark; reflectance set to 0 for: " + ", ".join(invalid_channels))

# ============================================================
# SERIAL CONNECTION
# ============================================================

print(f"Opening {SERIAL_PORT}...")

ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=1)

ser.write(b"l")

print("Connected to ESP32.")
print("Waiting for sensor data...")


# ============================================================
# MATPLOTLIB SETUP
# ============================================================

# Equal spacing for all 12 channels
x = list(range(len(WAVELENGTHS)))

fig, ax = plt.subplots(figsize=(12, 7))

bars = ax.bar(x, [0] * len(WAVELENGTHS), width=0.6, color=BAR_COLORS, edgecolor="black")

# Wavelength labels directly below each bar
ax.set_xticks(x)
ax.set_xticklabels([str(w) for w in WAVELENGTHS[:-1]] + ["855 (NIR)"])

ax.set_xlabel("Wavelength (nm)")
ax.set_ylabel("Reflectance")
ax.set_title("AS7343 Live Reflectance")

# Keep bars nicely inside the plot
ax.set_xlim(-0.5, len(WAVELENGTHS) - 0.5)

ax.grid(axis="y", linestyle="--", alpha=0.3)

ax.set_ylim(0, 1)

# ============================================================
# DATA STORAGE
# ============================================================

latest_values: list[float] = [0] * len(WAVELENGTHS)


# ============================================================
# SERIAL DATA READER
# ============================================================


def read_sensor_data() -> None:
    """Consume buffered live frames and retain the latest valid spectrum.

    Only "DATA" frames with N_CH finite raw counts in [0, 65535] are used.
    Valid frames replace the global latest_values with normalized reflectance.
    Malformed frames are skipped, and parse failures are printed.

    Raises:
        serial.SerialException: Reading the open serial connection fails.
    """
    global latest_values

    # Read everything currently waiting in the serial buffer
    while ser.in_waiting:
        line = ser.readline().decode("utf-8", errors="ignore").strip()

        if not line:
            continue

        # We only care about DATA lines
        if not line.startswith("DATA,"):
            continue

        try:
            parts = line.split(",")

            # First element is "DATA"
            values = parts[1:]

            # Firmware sends 12 raw spectral channels in CHANNEL_NAMES order.
            if len(values) != N_CH:
                continue

            raw_values = [float(value) for value in values]
            if not all(math.isfinite(value) and 0 <= value <= 65535 for value in raw_values):
                print("Invalid raw sensor values:", line)
                continue
            latest_values = normalize(raw_values)

        except ValueError:
            print("Could not parse:", line)


# ============================================================
# UPDATE GRAPH
# ============================================================


def update(frame: int) -> BarContainer:
    """Refresh reflectance bars from the latest buffered sensor data.

    Args:
        frame: Animation frame number supplied by Matplotlib; it is unused.

    Returns:
        Existing bar container after updating bar heights and the maximum
        reflectance shown in the plot title. No new plot is created.

    Raises:
        serial.SerialException: Reading sensor data fails.
    """
    read_sensor_data()

    max_value = max(latest_values)

    # Update bar heights
    for bar, value in zip(bars, latest_values):
        bar.set_height(value)

    # --------------------------------------------------------
    # Update title
    # --------------------------------------------------------

    ax.set_title(f"AS7343 Live Reflectance   Max = {max_value:.4f}")

    return bars


# ============================================================
# START ANIMATION
# ============================================================

animation = FuncAnimation(fig, update, interval=100, blit=False, cache_frame_data=False)

plt.tight_layout()

try:
    plt.show()

finally:
    try:
        ser.write(b"s")
    finally:
        ser.close()
    print("Serial connection closed.")
