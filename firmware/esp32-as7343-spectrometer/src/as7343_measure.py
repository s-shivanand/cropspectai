"""Acquire AS7343 readings over USB serial and save normalized measurements.

Importing this module does not open the device. Run it as a script for the
interactive label prompt, or initialize its references and serial connection
before using the acquisition helpers. Keep SENSOR_SETTINGS synchronized
with main.cpp. Calibration references are raw counts, not reflectance.
"""

import csv
import math
import os
import time
from collections.abc import Callable, Sequence
from datetime import datetime
from pathlib import Path
from statistics import mean, stdev

import serial

import fimware_config
from fimware_config import CHANNEL_NAMES, N_CH, WAVELENGTHS  # noqa: F401

# ============================================================
# SETTINGS
# ============================================================

DARK_REF: list[float]
WHITE_REF: list[float]
ser: serial.Serial

SERIAL_PORT: str = "/dev/cu.usbserial-0001"
BAUD_RATE: int = 115200
PATH: Path = Path(__file__).resolve().parent.parent / "data"
CSV_FILE: Path = fimware_config.MISC_DATA_DIR / "reflectance.csv"
# CSV_FILE = fimware_config.EXPERIMENT_DATA_DIR /"basil_experiment.csv"
REF_FILE: Path = fimware_config.REF_DATA_DIR / "REF.csv"

# Sensor settings used by the firmware. Keep these in sync with main.cpp,
# they are saved in every CSV row so you always know how a row was measured.
SENSOR_SETTINGS: dict[str, int | str] = {
    "led_ma": 20,
    "gain": "64X",
    "atime": 29,
    "astep": 599,
    "n_avg": 10,
}

# Warn if a channel's standard deviation exceeds this fraction of its mean
NOISE_WARN_FRACTION: float = 0.05


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
    """Normalize one raw reading to reflectance using loaded references.

    Args:
        raw_values: Raw channel counts in CHANNEL_NAMES order.

    Returns:
        Reflectance values clamped to [0, 1]. A channel with a white reference
        at or below its dark reference produces zero.

    Raises:
        ValueError: The reading does not contain N_CH values.

    Note:
        Requires DARK_REF and WHITE_REF to have been initialized by the caller
        or interactive measurement entry point.
    """
    if len(raw_values) != N_CH:
        raise ValueError(f"Expected {N_CH} channel values")
    return [
        max(0.0, min(1.0, (raw - dark) / (white - dark))) if white > dark else 0.0
        for raw, dark, white in zip(raw_values, DARK_REF, WHITE_REF)
    ]


HEADER: list[str] = (
    ["timestamp", "specimen"]
    + [f"{n}" for n in CHANNEL_NAMES]
    + ["noisy_channels"]
    + list(SENSOR_SETTINGS.keys())
)

# ============================================================
# MEASUREMENT
# ============================================================


def take_measurement(timeout_s: float = 15) -> list[list[float]] | None:
    """Request and collect a complete batch of raw sensor readings.

    Args:
        timeout_s: Maximum batch collection time in seconds, checked between
            serial reads. Individual reads use the connection's own timeout.

    Returns:
        List of SENSOR_SETTINGS["n_avg"] readings, each with N_CH channel values,
        or None on timeout, malformed data, an incomplete batch, or a reported
        sensor read failure. Valid values are finite raw counts in [0, 65535].

    Raises:
        serial.SerialException: Serial communication fails.

    Note:
        Uses the initialized global serial connection, clears its input buffer,
        and sends the firmware's "r" command. Failure details are printed.
    """
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


def normalized_statistics(
    raw_readings: Sequence[Sequence[float]],
) -> tuple[list[float], list[float]]:
    """Normalize each reading, then compute per-channel batch statistics.

    Args:
        raw_readings: Batch of SENSOR_SETTINGS["n_avg"] readings, with channels
            ordered as CHANNEL_NAMES in every reading.

    Returns:
        Pair of lists containing mean reflectance and sample standard deviation
        per channel. Normalization and clamping happen before aggregation.

    Raises:
        ValueError: The batch size or a reading's channel count is incorrect.
        statistics.StatisticsError: Fewer than two readings are available for
            sample standard deviation.
    """
    if len(raw_readings) != SENSOR_SETTINGS["n_avg"]:
        raise ValueError(f"Expected {SENSOR_SETTINGS['n_avg']} readings")
    normalized = [normalize(reading) for reading in raw_readings]
    channels = list(zip(*normalized))
    return [mean(values) for values in channels], [stdev(values) for values in channels]


def check_noise(means: Sequence[float], stds: Sequence[float]) -> list[str]:
    """Describe channels whose relative standard deviation is too high.

    Args:
        means: Per-channel mean reflectance values in CHANNEL_NAMES order.
        stds: Per-channel sample standard deviations in the same order.

    Returns:
        List of channel names and percentage noise, such as "F1 (8.0%)".
        A channel is flagged only when its mean is positive and its standard
        deviation divided by its mean exceeds NOISE_WARN_FRACTION.
    """
    noisy = []
    for name, m, s in zip(CHANNEL_NAMES, means, stds):
        if m > 0 and s / m > NOISE_WARN_FRACTION:
            noisy.append(f"{name} ({100 * s / m:.1f}%)")
    return noisy


# ============================================================
# MAIN LOOP
# ============================================================


def main(
    on_measurement: Callable[[dict[str, str | float | int]], object] | None = None,
    csv_file: str | Path | None = None,
) -> None:
    """Run the interactive serial measurement and CSV-saving session.

    Args:
        on_measurement: Optional callback receiving each saved measurement as
            a dictionary. Its return value is ignored; callback errors propagate.
        csv_file: Destination CSV, or None to use the configured CSV_FILE.
            Parent directories and a missing file are created automatically.

    Raises:
        FileNotFoundError: The reference CSV is missing.
        ValueError: References or an existing measurement CSV header are invalid.
        serial.SerialException: The serial connection cannot be opened or used.
        OSError: Measurement output cannot be created or written.

    Note:
        Initializes global references and the serial connection. A specimen
        label starts a measurement, Enter reuses the label, and "q" ends the
        session. Saved rows contain timestamp, specimen, reflectance, noise
        labels, and sensor settings. The callback runs after a row is flushed.
        The measurement loop's cleanup stops streaming and closes its handles.
    """
    global DARK_REF, WHITE_REF, ser
    csv_file = Path(csv_file) if csv_file is not None else CSV_FILE

    DARK_REF, WHITE_REF = load_references(REF_FILE)
    print(f"Loaded dark and white references from {REF_FILE}")
    invalid_channels = [
        name for name, dark, white in zip(CHANNEL_NAMES, DARK_REF, WHITE_REF) if white <= dark
    ]
    if invalid_channels:
        print("Warning: white <= dark; reflectance set to 0 for: " + ", ".join(invalid_channels))

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
        print(
            "Did not see AS7343_READY. Continuing anyway, "
            "but if measurements fail, press the RST button and restart this script."
        )

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
                print(
                    "  Warning, noisy channels (std > "
                    f"{int(NOISE_WARN_FRACTION * 100)}% of mean): " + ", ".join(noisy)
                )
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
