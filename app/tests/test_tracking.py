"""Check specimen paths and append-only CSV tracking."""

import csv
import tempfile
import unittest
from pathlib import Path

from app.tracking import (
    append_tracking_prediction,
    read_tracking_predictions,
    specimen_tracking_path,
)


class TrackingTests(unittest.TestCase):
    """Check safe specimen filenames and append-only tracking behavior."""

    def test_specimen_names_are_safe_and_distinct(self) -> None:
        """Verify specimen names are safe and distinct."""
        prediction_file = Path("/tmp/predictions.csv")
        ordinary = specimen_tracking_path("leaf_01", prediction_file)
        self.assertEqual(ordinary.name, "leaf_01.csv")
        for label in ("../leaf", "leaf/name", ".", "leaf" * 100):
            path = specimen_tracking_path(label, prediction_file)
            self.assertEqual(path.parent, ordinary.parent)
            self.assertEqual(path.suffix, ".csv")
        self.assertNotEqual(
            specimen_tracking_path("leaf/name", prediction_file),
            specimen_tracking_path("leaf_name", prediction_file),
        )

    def test_rows_append_with_the_same_values_and_columns(self) -> None:
        """Verify rows append with the same values and columns."""
        with tempfile.TemporaryDirectory() as directory:
            path = specimen_tracking_path("leaf_01", Path(directory) / "predictions.csv")
            row = {
                "timestamp": "2026-10-16T16:00:00",
                "specimen": "leaf_01",
                "FZ": 0.1,
                "mndvi_as7343": 1.5,
                "cab": 34.25,
            }
            append_tracking_prediction(row, path)
            append_tracking_prediction({**row, "timestamp": "2026-10-17T16:00:00"}, path)
            with path.open(newline="") as handle:
                reader = csv.DictReader(handle)
                self.assertEqual(reader.fieldnames, list(row))
                rows = list(reader)
            self.assertEqual(len(rows), 2)
            self.assertEqual(float(rows[0]["mndvi_as7343"]), 1.5)
            self.assertEqual(float(rows[0]["cab"]), 34.25)

    def test_schema_or_specimen_mismatch_does_not_change_file(self) -> None:
        """Verify schema or specimen mismatch does not change file."""
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "leaf.csv"
            row = {"timestamp": "now", "specimen": "leaf", "cab": 34.25}
            append_tracking_prediction(row, path)
            before = path.read_bytes()
            with self.assertRaises(ValueError):
                append_tracking_prediction({**row, "new_feature": 1}, path)
            with self.assertRaises(ValueError):
                append_tracking_prediction({**row, "specimen": "Leaf"}, path)
            self.assertEqual(path.read_bytes(), before)

    def test_semicolon_file_keeps_format_and_quoted_values(self) -> None:
        """Verify semicolon file keeps format and quoted values."""
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "sun_no_water.csv"
            row = {
                "timestamp": "2026-10-18T10:00:00",
                "specimen": "sun_no_water",
                "noisy_channels": "FZ (6.5%); F8 (5.4%)",
                "cab": 34.25,
            }
            with path.open("w", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(row), delimiter=";")
                writer.writeheader()
                writer.writerow(row)
            path.write_bytes(path.read_bytes().rstrip(b"\r\n"))
            before = path.read_bytes()
            append_tracking_prediction({**row, "cab": 35.5}, path)
            self.assertTrue(path.read_bytes().startswith(before))
            rows = read_tracking_predictions(path)
            self.assertEqual(len(rows), 2)
            self.assertEqual(rows[0]["noisy_channels"], row["noisy_channels"])
            self.assertEqual(rows[1]["cab"], "35.5")
            before = path.read_bytes()
            with self.assertRaisesRegex(ValueError, "different feature columns"):
                append_tracking_prediction({**row, "new_feature": 1}, path)
            with self.assertRaisesRegex(ValueError, "different specimen label"):
                append_tracking_prediction({**row, "specimen": "other"}, path)
            self.assertEqual(path.read_bytes(), before)


if __name__ == "__main__":
    unittest.main()
