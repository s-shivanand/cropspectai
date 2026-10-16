"""Append exact prediction rows to persistent, specimen-specific CSV files."""

import csv
import hashlib
import re
from pathlib import Path
from threading import Lock

_TRACKING_LOCK = Lock()


def read_tracking_predictions(path):
    """Read a complete snapshot without racing a specimen CSV append."""
    with _TRACKING_LOCK:
        with Path(path).open(newline="") as handle:
            return list(csv.DictReader(handle))


def specimen_tracking_path(specimen, prediction_file):
    label = str(specimen).strip()
    if not label:
        raise ValueError("A specimen label is required for tracking.")
    filename = re.sub(r"[^\w.-]", "_", label).strip("._")[:100]
    if filename != label:
        suffix = hashlib.sha256(label.encode("utf-8")).hexdigest()[:10]
        filename = f"{filename or 'specimen'}_{suffix}"
    return Path(prediction_file).parent / "tracked_specimens" / f"{filename}.csv"


def append_tracking_prediction(row, path):
    """Preserve the usual prediction CSV's values, header, and column order."""
    if row is None:
        raise ValueError("Take a new measurement before enabling tracking.")
    columns = list(row)
    if not {"timestamp", "specimen", "cab"}.issubset(columns):
        raise ValueError("A complete saved prediction is required for tracking.")
    path = Path(path)
    with _TRACKING_LOCK:
        new_file = not path.exists() or path.stat().st_size == 0
        if not new_file:
            with path.open(newline="") as handle:
                reader = csv.DictReader(handle)
                if reader.fieldnames != columns:
                    raise ValueError("The specimen tracking file has different feature columns.")
                first_row = next(reader, None)
                if first_row is not None and first_row["specimen"] != row["specimen"]:
                    raise ValueError("This tracking file belongs to a different specimen label.")
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=columns)
            if new_file:
                writer.writeheader()
            writer.writerow(row)
