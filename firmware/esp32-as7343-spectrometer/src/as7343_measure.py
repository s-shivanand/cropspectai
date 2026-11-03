import csv
import os
import math
import time
from statistics import mean, stdev
from datetime import datetime
from pathlib import Path

import serial

import fimware_config
from fimware_config import CHANNEL_NAMES, WAVELENGTHS, N_CH

# ============================================================
# SETTINGS
# ============================================================

SERIAL_PORT = "/dev/cu.usbserial-0001"
BAUD_RATE = 115200
PATH = Path(__file__).resolve().parent.parent / "data"
CSV_FILE = fimware_config.MISC_DATA_DIR / "reflectance.csv"
# CSV_FILE = fimware_config.RAW_DATA_DIR /"basil_experiment.csv"
REF_FILE = fimware_config.REF_DATA_DIR / "REF.csv"

# Sensor settings used by the firmware. Keep these in sync with main.cpp,
# they are saved in every CSV row so you always know how a row was measured.
SENSOR_SETTINGS = {
    "led_ma": 20,
    "gain": "64X",
    "atime": 29,
    "astep": 599,
    "n_avg": 10,
}

# Warn if a channel's standard deviation exceeds this fraction of its mean
NOISE_WARN_FRACTION = 0.05

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


HEADER = (
    ["timestamp", "specimen"]
    + [f"{n}" for n in CHANNEL_NAMES]
    + ["noisy_channels"]
    + list(SENSOR_SETTINGS.keys())
)

# ============================================================
# MEASUREMENT
# ============================================================

def take_measurement(timeout_s=15):
    """Collect a complete raw batch; return readings or None on failure."""
    ser.reset_input_buffer()
    ser.write(b"r")
    readings = []

    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        line = ser.readline().decode("utf-8", errors="ignore").strip()
        if not line:
            continue

        if line == "RAW_DONE":
            if len(readings) != SENSOR_SETTINGS["n_avg"]:
                print(f"Incomplete raw batch: received {len(readings)} readings.")
                return None
            return readings

        if line.startswith("RAW,"):
            parts = line.split(",")[1:]
            if len(parts) != N_CH:
                print(f"Unexpected RAW line length ({len(parts)} values): {line}")
                return None
            try:
                values = [float(p) for p in parts]
            except ValueError:
                print("Could not parse:", line)
                return None
            if not all(math.isfinite(value) and 0 <= value <= 65535 for value in values):
                print("Invalid raw sensor values:", line)
                return None
            readings.append(values)
            if len(readings) > SENSOR_SETTINGS["n_avg"]:
                print("Too many readings in raw batch.")
                return None

        if line.startswith("Read failed"):
            print("ESP32 reported:", line)
            return None

    print("Timed out waiting for raw readings. Upload the firmware supporting command 'r'.")
    return None


def normalized_statistics(raw_readings):
    """Normalize each reading first, then compute per-channel sample statistics."""
    if len(raw_readings) != SENSOR_SETTINGS["n_avg"]:
        raise ValueError(f"Expected {SENSOR_SETTINGS['n_avg']} readings")
    normalized = [normalize(reading) for reading in raw_readings]
    channels = list(zip(*normalized))
    return [mean(values) for values in channels], [stdev(values) for values in channels]


def check_noise(means, stds):
    noisy = []
    for name, m, s in zip(CHANNEL_NAMES, means, stds):
        if m > 0 and s / m > NOISE_WARN_FRACTION:
            noisy.append(f"{name} ({100 * s / m:.1f}%)")
    return noisy


# ============================================================
# MAIN LOOP
# ============================================================


def main(on_measurement=None, csv_file=None):
    """Run interactive measurements, optionally handling each saved CSV row."""
    global DARK_REF, WHITE_REF, ser
    csv_file = Path(csv_file) if csv_file is not None else CSV_FILE

    DARK_REF, WHITE_REF = load_references(REF_FILE)
    print(f"Loaded dark and white references from {REF_FILE}")
    invalid_channels = [
        name for name, dark, white in zip(CHANNEL_NAMES, DARK_REF, WHITE_REF)
        if white <= dark
    ]
    if invalid_channels:
        print("Warning: white <= dark; reflectance set to 0 for: "
              + ", ".join(invalid_channels))

    csv_file.parent.mkdir(parents=True, exist_ok=True)
    new_file = (not os.path.exists(csv_file)) or os.path.getsize(csv_file) == 0
    if not new_file:
        with open(csv_file, newline="") as handle:
            if next(csv.reader(handle), None) != HEADER:
                raise ValueError(f"Existing CSV header does not match: {csv_file}")
    csv_handle = open(csv_file, "a", newline="")
    writer = csv.writer(csv_handle)
    if new_file:
        writer.writerow(HEADER)
        csv_handle.flush()

    # ============================================================
    # SERIAL CONNECTION
    # ============================================================

    print(f"Opening {SERIAL_PORT}...")
    ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=1)

    # Opening the port usually resets the ESP32, so wait for it to boot
    print("Waiting for ESP32 to be ready (about 5 seconds)...")
    time.sleep(2)
    deadline = time.time() + 5
    ready = False
    while time.time() < deadline:
        line = ser.readline().decode("utf-8", errors="ignore").strip()
        if line == "AS7343_READY":
            ready = True
            break

    if not ready:
        print("Did not see AS7343_READY. Continuing anyway, "
              "but if measurements fail, press the RST button and restart this script.")

    # Make sure live streaming is off and the buffer is clean
    ser.write(b"s")
    time.sleep(0.3)
    ser.reset_input_buffer()

    print("\nReady.")
    print("Commands at the label prompt:")
    print("  type a label  -> use it for the next measurement (e.g. leaf_01)")
    print("  just Enter    -> reuse the previous label")
    print("  q             -> quit\n")

    last_label = ""
    count = 0

    try:
        while True:
            prompt = f"Label [{last_label}]: " if last_label else "Label: "
            entry = input(prompt).strip()

            if entry.lower() == "q":
                break
            label = entry if entry else last_label
            if not label:
                print("Please enter a label first.")
                continue

            print("Measuring...")
            result = take_measurement()
            if result is None:
                print("Measurement failed, nothing saved. Try again.\n")
                continue

            reflectance, reflectance_stds = normalized_statistics(result)
            noisy = check_noise(reflectance, reflectance_stds)
            last_label = label

            row = (
                [datetime.now().isoformat(timespec="seconds"), label]
                + [round(r, 6) for r in reflectance]
                + ["; ".join(noisy)]
                + list(SENSOR_SETTINGS.values())
            )
            writer.writerow(row)
            csv_handle.flush()
            count += 1

            print(f"Saved #{count} ({label}):")
            print("  " + "  ".join(f"{n}={r:.4f}" for n, r in zip(CHANNEL_NAMES, reflectance)))

            if noisy:
                print("  Warning, noisy channels (std > "
                      f"{int(NOISE_WARN_FRACTION * 100)}% of mean): " + ", ".join(noisy))
            if max(value for reading in result for value in reading) > 60000:
                print("  Warning, a channel is near saturation. Lower the gain or LED current.")
            if on_measurement is not None:
                on_measurement(dict(zip(HEADER, row)))
            print()

    finally:
        ser.write(b"s")
        ser.close()
        csv_handle.close()
        print(f"Done. {count} measurements saved to {csv_file}")


if __name__ == "__main__":
    main()
