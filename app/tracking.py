"""Append exact prediction rows to persistent, specimen-specific CSV files."""

import csv
import hashlib
import re
from pathlib import Path
from threading import Lock

_TRACKING_LOCK = Lock()


def _tracking_delimiter(handle):
    """Recognize comma CSVs and semicolon CSVs saved by spreadsheet tools."""
    header = handle.readline()
    handle.seek(0)
    try:
        return csv.Sniffer().sniff(header, delimiters=",;").delimiter
    except csv.Error:
        return ","


def read_tracking_predictions(path):
    """Read a complete snapshot without racing a specimen CSV append."""
    with _TRACKING_LOCK:
        with Path(path).open(newline="") as handle:
            return list(csv.DictReader(handle, delimiter=_tracking_delimiter(handle)))


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
        delimiter = ","
        needs_newline = False
        if not new_file:
            with path.open(newline="") as handle:
                delimiter = _tracking_delimiter(handle)
                reader = csv.DictReader(handle, delimiter=delimiter)
                if reader.fieldnames != columns:
                    raise ValueError("The specimen tracking file has different feature columns.")
                first_row = next(reader, None)
                if first_row is not None and first_row["specimen"] != row["specimen"]:
                    raise ValueError("This tracking file belongs to a different specimen label.")
            with path.open("rb") as handle:
                handle.seek(-1, 2)
                needs_newline = handle.read(1) not in (b"\r", b"\n")
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=columns, delimiter=delimiter)
            if new_file:
                writer.writeheader()
            elif needs_newline:
                handle.write("\r\n")
            writer.writerow(row)
