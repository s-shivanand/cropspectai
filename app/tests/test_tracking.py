"""Check specimen paths and append-only CSV tracking."""

import csv
import tempfile
import unittest
from pathlib import Path

from app.tracking import append_tracking_prediction, specimen_tracking_path


class TrackingTests(unittest.TestCase):
    def test_specimen_names_are_safe_and_distinct(self):
        prediction_file = Path("/tmp/predictions.csv")
        ordinary = specimen_tracking_path("leaf_01", prediction_file)
        self.assertEqual(ordinary.name, "leaf_01.csv")
        for label in ("../leaf", "leaf/name", ".", "leaf" * 100):
            path = specimen_tracking_path(label, prediction_file)
            self.assertEqual(path.parent, ordinary.parent)
            self.assertEqual(path.suffix, ".csv")
        self.assertNotEqual(specimen_tracking_path("leaf/name", prediction_file),
                            specimen_tracking_path("leaf_name", prediction_file))

    def test_rows_append_with_the_same_values_and_columns(self):
        with tempfile.TemporaryDirectory() as directory:
            path = specimen_tracking_path("leaf_01", Path(directory) / "predictions.csv")
            row = {"timestamp": "2026-10-16T16:00:00", "specimen": "leaf_01",
                   "FZ": 0.1, "mndvi_as7343": 1.5, "cab": 34.25}
            append_tracking_prediction(row, path)
            append_tracking_prediction({**row, "timestamp": "2026-10-17T16:00:00"}, path)
            with path.open(newline="") as handle:
                reader = csv.DictReader(handle)
                self.assertEqual(reader.fieldnames, list(row))
                rows = list(reader)
            self.assertEqual(len(rows), 2)
            self.assertEqual(float(rows[0]["mndvi_as7343"]), 1.5)
            self.assertEqual(float(rows[0]["cab"]), 34.25)

    def test_schema_or_specimen_mismatch_does_not_change_file(self):
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


if __name__ == "__main__":
    unittest.main()
