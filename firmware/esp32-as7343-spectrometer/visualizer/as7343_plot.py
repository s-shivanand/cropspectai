import csv
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from fimware_config import CHANNEL_NAMES, WAVELENGTHS, BAR_COLORS, N_CH

import serial
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation


# ============================================================
# SETTINGS
# ============================================================

SERIAL_PORT = "/dev/cu.usbserial-0001"   # CHANGE THIS
BAUD_RATE = 115200
REF_FILE = Path(__file__).resolve().parent.parent / "data" / "ref" / "REF.csv"

def load_references(ref_file):
    """Load raw channel means in CHANNEL_NAMES order, regardless of CSV order."""
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


def normalize(raw_values):
    """Convert one raw reading to reflectance, clamped to 0–1."""
    if len(raw_values) != N_CH:
        raise ValueError(f"Expected {N_CH} channel values")
    return [
        max(0.0, min(1.0, (raw - dark) / (white - dark)))
        if white > dark else 0.0
        for raw, dark, white in zip(raw_values, DARK_REF, WHITE_REF)
    ]


DARK_REF, WHITE_REF = load_references(REF_FILE)
print(f"Loaded dark and white references from {REF_FILE}")
invalid_channels = [
    name for name, dark, white in zip(CHANNEL_NAMES, DARK_REF, WHITE_REF)
    if white <= dark
]
if invalid_channels:
    print("Warning: white <= dark; reflectance set to 0 for: "
          + ", ".join(invalid_channels))

# ============================================================
# SERIAL CONNECTION
# ============================================================

print(f"Opening {SERIAL_PORT}...")

ser = serial.Serial(
    SERIAL_PORT,
    BAUD_RATE,
    timeout=1
)

ser.write(b'l')

print("Connected to ESP32.")
print("Waiting for sensor data...")


# ============================================================
# MATPLOTLIB SETUP
# ============================================================

# Equal spacing for all 12 channels
x = list(range(len(WAVELENGTHS)))

fig, ax = plt.subplots(figsize=(12, 7))

bars = ax.bar(
    x,
    [0] * len(WAVELENGTHS),
    width=0.6,
    color=BAR_COLORS,
    edgecolor="black"
)

# Wavelength labels directly below each bar
ax.set_xticks(x)
ax.set_xticklabels(
    [str(w) for w in WAVELENGTHS[:-1]] + ["855 (NIR)"]
)

ax.set_xlabel("Wavelength (nm)")
ax.set_ylabel("Reflectance")
ax.set_title("AS7343 Live Reflectance")

# Keep bars nicely inside the plot
ax.set_xlim(-0.5, len(WAVELENGTHS) - 0.5)

ax.grid(
    axis="y",
    linestyle="--",
    alpha=0.3
)

ax.set_ylim(0, 1)

# ============================================================
# DATA STORAGE
# ============================================================

latest_values = [0] * len(WAVELENGTHS)


# ============================================================
# SERIAL DATA READER
# ============================================================

def read_sensor_data():

    global latest_values

    # Read everything currently waiting in the serial buffer
    while ser.in_waiting:

        line = ser.readline().decode(
            "utf-8",
            errors="ignore"
        ).strip()

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
            if not all(math.isfinite(value) and 0 <= value <= 65535
                       for value in raw_values):
                print("Invalid raw sensor values:", line)
                continue
            latest_values = normalize(raw_values)

        except ValueError:
            print("Could not parse:", line)


# ============================================================
# UPDATE GRAPH
# ============================================================

def update(frame):

    read_sensor_data()

    max_value = max(latest_values)

    # Update bar heights
    for bar, value in zip(bars, latest_values):
        bar.set_height(value)

    # --------------------------------------------------------
    # Update title
    # --------------------------------------------------------

    ax.set_title(
        "AS7343 Live Reflectance   "
        f"Max = {max_value:.4f}"
    )

    return bars


# ============================================================
# START ANIMATION
# ============================================================

animation = FuncAnimation(
    fig,
    update,
    interval=100,
    blit=False,
    cache_frame_data=False
)

plt.tight_layout()

try:
    plt.show()

finally:
    try:
        ser.write(b"s")
    finally:
        ser.close()
    print("Serial connection closed.")
