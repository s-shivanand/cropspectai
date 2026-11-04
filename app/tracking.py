"""Append exact prediction rows to persistent, specimen-specific CSV files."""

import csv
import hashlib
import re
from collections.abc import Mapping
from pathlib import Path
from threading import Lock
from typing import TextIO

_TRACKING_LOCK = Lock()


def _tracking_delimiter(handle: TextIO) -> str:
    """Detect comma or semicolon delimiters and rewind the input handle.

    Args:
        handle: Seekable text stream positioned at the start of the CSV header.

    Returns:
        Detected delimiter, falling back to a comma if detection fails. The
        stream is positioned at zero on return.
    """
    header = handle.readline()
    handle.seek(0)
    try:
        return csv.Sniffer().sniff(header, delimiters=",;").delimiter
    except csv.Error:
        return ","


def read_tracking_predictions(path: str | Path) -> list[dict[str, str]]:
    """Read a snapshot of specimen tracking rows under the append lock.

    Args:
        path: Comma- or semicolon-delimited tracking CSV to read.

    Returns:
        List of row dictionaries containing the recorded field strings.

    Raises:
        OSError: The file cannot be opened or read.
    """
    with _TRACKING_LOCK:
        with Path(path).open(newline="") as handle:
            return list(csv.DictReader(handle, delimiter=_tracking_delimiter(handle)))


def specimen_tracking_path(specimen: str, prediction_file: str | Path) -> Path:
    """Build a safe specimen-specific CSV path beside the prediction file.

    Args:
        specimen: Nonempty specimen label; surrounding whitespace is removed.
        prediction_file: Central prediction CSV used to locate the parent folder.

    Returns:
        Path under the prediction file's tracked_specimens directory. Unsafe
        or truncated labels receive a hash suffix to distinguish filenames.

    Raises:
        ValueError: The stripped specimen label is empty.

    Note:
        Only computes a path; no directories or files are created.
    """
    label = str(specimen).strip()
    if not label:
        raise ValueError("A specimen label is required for tracking.")
    filename = re.sub(r"[^\w.-]", "_", label).strip("._")[:100]
    if filename != label:
        suffix = hashlib.sha256(label.encode("utf-8")).hexdigest()[:10]
        filename = f"{filename or 'specimen'}_{suffix}"
    return Path(prediction_file).parent / "tracked_specimens" / f"{filename}.csv"


def append_tracking_prediction(row: Mapping[str, object] | None, path: str | Path) -> None:
    """Append an exact prediction row to its specimen tracking CSV.

    Args:
        row: Ordered mapping of saved prediction columns and values, including
            timestamp, specimen, and cab. None indicates no saved prediction.
        path: Destination specimen tracking CSV.

    Raises:
        ValueError: The row is incomplete, or an existing file's column order
            or specimen label differs from the supplied row.
        OSError: The file cannot be read or written.

    Note:
        Creates parent directories and the header when needed. Existing comma
        or semicolon delimiters are preserved. Reads and appends share a lock.
    """
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
