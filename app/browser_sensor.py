"""Bridge the existing app controls to Chrome's local USB serial connection.

Only acquisition moves to the browser. Calibration, model inference, CSV output,
and tracking retain the existing Python workflow. No serial port is shared
between browser sessions, and no new controls are rendered.
"""

import csv
import math
from collections.abc import Sequence
from datetime import datetime
from pathlib import Path
from statistics import mean, stdev

import streamlit as st
import streamlit.components.v2 as components

from app.live_sensor import LiveReading
from scripts import measure_predict as workflow

_bridge = components.component(
    "browser_serial_bridge",
    js=Path(__file__).with_name("browser_serial.js").read_text(),
    isolate_styles=False,
)


def browser_sensor(session: str) -> dict:
    """Mount an invisible serial bridge beside the existing device controls.

    Args:
        session: Current sensor-session identifier; reset creates a new bridge.

    Returns:
        Latest browser packet, or an empty mapping before the frontend starts.
        Unsupported browsers retain the existing local Python serial workflow.
    """
    result = _bridge(
        key=f"browser_serial_{session}",
        data={
            "channels": workflow.fimware_config.N_CH,
            "count": workflow.as7343_measure.SENSOR_SETTINGS["n_avg"],
            "baud": workflow.as7343_measure.BAUD_RATE,
            "ack": st.session_state.get("browser_saved_id"),
        },
        on_packet_change=lambda: None,
    )
    return result.packet or {}


def validate_reading(values: object) -> tuple[float, ...]:
    """Validate one browser-supplied frame before calibration or file access.

    Args:
        values: Raw sensor counts in configured channel order.

    Returns:
        Finite counts between zero and 65535, inclusive.

    Raises:
        ValueError: The frame has invalid dimensions, types, or counts.
    """
    if not isinstance(values, (list, tuple)) or len(values) != workflow.fimware_config.N_CH:
        raise ValueError("The device returned an invalid channel count.")
    if any(
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
        or not 0 <= value <= 65535
        for value in values
    ):
        raise ValueError("The device returned invalid sensor counts.")
    return tuple(float(value) for value in values)


def normalize_reading(
    values: Sequence[float], references: tuple[list[float], list[float]]
) -> tuple[float, ...]:
    """Apply the firmware helper's dark/white calibration without global state.

    Args:
        values: Validated raw channel counts.
        references: Dark and white references loaded from the configured REF.csv.

    Returns:
        Reflectance clamped to [0, 1], with invalid reference ranges set to zero.
    """
    dark, white = references
    return tuple(
        max(0.0, min(1.0, (value - low) / (high - low))) if high > low else 0.0
        for value, low, high in zip(values, dark, white)
    )


def browser_reading(packet: dict) -> LiveReading:
    """Translate browser status and live counts into the existing preview model.

    Args:
        packet: Current bridge state, including optional raw live counts.

    Returns:
        Calibrated preview and acquisition errors using the existing UI model.
    """
    try:
        references = workflow.as7343_measure.load_references(workflow.as7343_measure.REF_FILE)
        values = packet.get("values")
        raw = validate_reading(values) if values is not None else None
        normalized = normalize_reading(raw, references) if raw is not None else None
        invalid = [
            name for name, dark, white in zip(workflow.CHANNELS, *references) if white <= dark
        ]
        warnings = (
            ("White reference <= dark reference; reflectance set to 0 for: " + ", ".join(invalid),)
            if invalid
            else ()
        )
        if raw is not None and max(raw) > 60000:
            warnings += ("A channel is near saturation. Lower the gain or LED current.",)
        return LiveReading(
            bool(packet.get("live")),
            normalized,
            packet.get("error") if not packet.get("request") else None,
            warnings,
        )
    except (OSError, ValueError) as exc:
        return LiveReading(error=str(exc))


def save_browser_measurement(raw: object, specimen: str) -> workflow.MeasurementResult:
    """Calibrate a complete browser batch and use the existing prediction writer.

    Args:
        raw: Ten raw readings in configured channel order.
        specimen: Specimen label entered in the existing app form.

    Returns:
        Saved measurement and prediction, including the existing quality warnings.

    Raises:
        ValueError: The label, batch, calibration, or existing CSV schema is invalid.
        OSError: Calibration or output files cannot be accessed.
        RuntimeError: Another measurement is writing shared output files.

    Note:
        Validation precedes file creation. As in the local workflow, a model
        failure preserves the measurement and returns a prediction error.
    """
    specimen = specimen.strip()
    if not specimen:
        raise ValueError("Please enter a specimen label before measuring.")
    sensor = workflow.as7343_measure
    if not isinstance(raw, (list, tuple)) or len(raw) != sensor.SENSOR_SETTINGS["n_avg"]:
        raise ValueError(f"Expected {sensor.SENSOR_SETTINGS['n_avg']} complete readings.")
    readings = [validate_reading(row) for row in raw]
    references = sensor.load_references(sensor.REF_FILE)
    normalized = [normalize_reading(row, references) for row in readings]
    columns = list(zip(*normalized))
    means = [mean(column) for column in columns]
    noisy = sensor.check_noise(means, [stdev(column) for column in columns])
    warnings = list(browser_reading({}).warnings)
    if noisy:
        warnings.append("Noisy channels: " + "; ".join(noisy))
    if max(max(row) for row in readings) > 60000:
        warnings.append("A channel is near saturation. Lower the gain or LED current.")
    row = (
        [datetime.now().isoformat(timespec="seconds"), specimen]
        + [round(value, 6) for value in means]
        + ["; ".join(noisy)]
        + list(sensor.SENSOR_SETTINGS.values())
    )
    measurement = dict(zip(sensor.HEADER, row))
    with workflow._exclusive_measurement():
        new_file = not workflow.CSV_FILE.exists() or workflow.CSV_FILE.stat().st_size == 0
        if not new_file:
            with workflow.CSV_FILE.open(newline="") as handle:
                if next(csv.reader(handle), None) != sensor.HEADER:
                    raise ValueError(f"Existing CSV header does not match: {workflow.CSV_FILE}")
        with workflow._prediction_output() as (features, writer, prediction_handle):
            workflow.CSV_FILE.parent.mkdir(parents=True, exist_ok=True)
            with workflow.CSV_FILE.open("a", newline="") as handle:
                measurement_writer = csv.writer(handle)
                if new_file:
                    measurement_writer.writerow(sensor.HEADER)
                measurement_writer.writerow(row)
            result = workflow._save_prediction(measurement, features, writer, prediction_handle)
            result.warnings = warnings
            return result
