"""Check that history uses saved values and handles unavailable observations."""

import csv
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from app.tracking_history import load_tracking_history


class TrackingHistoryTests(unittest.TestCase):
    """Check history ordering, saved values, and unavailable observations."""

    def test_sorted_saved_values_and_unavailable_points(self) -> None:
        """Verify sorted saved values and unavailable points."""
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "leaf.csv"
            with path.open("w", newline="") as handle:
                writer = csv.writer(handle)
                writer.writerow(["timestamp", "specimen", "cab", "mndvi_as7343"])
                writer.writerows(
                    [
                        ["2026-10-17T10:00:00", "leaf", 45, 1.5],
                        ["2026-10-16T10:00:00", "leaf", 30, -0.2],
                        ["2026-10-18T10:00:00", "leaf", 40, "inf"],
                        ["not a timestamp", "leaf", 42, 0.8],
                    ]
                )
            original = path.read_bytes()
            data = load_tracking_history(path, "leaf")
            self.assertEqual(data["cab"].tolist(), [30, 45, 40])
            self.assertEqual(data["mndvi_as7343"].iloc[:2].tolist(), [-0.2, 1.5])
            self.assertTrue(pd.isna(data["mndvi_as7343"].iloc[2]))
            self.assertEqual(path.read_bytes(), original)
            with self.assertRaisesRegex(ValueError, "different specimen"):
                load_tracking_history(path, "other")

    def test_older_file_without_mndvi_keeps_cab_history(self) -> None:
        """Verify older file without MNDVI keeps Cab history."""
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "leaf.csv"
            path.write_text("timestamp,specimen,cab\n2026-10-16T10:00:00,leaf,34.5\n")
            data = load_tracking_history(path, "leaf")
            self.assertEqual(data["cab"].tolist(), [34.5])
            self.assertTrue(data["mndvi_as7343"].isna().all())

    def test_semicolon_file_loads_history(self) -> None:
        """Verify semicolon file loads history."""
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "sun_no_water.csv"
            path.write_text(
                "timestamp;specimen;cab;mndvi_as7343;noisy_channels\n"
                '2026-10-18T10:00:00;sun_no_water;34.5;0.8;"FZ (6.5%); F8 (5.4%)"\n'
            )
            data = load_tracking_history(path, "sun_no_water")
            self.assertEqual(data["cab"].tolist(), [34.5])
            self.assertEqual(data["mndvi_as7343"].tolist(), [0.8])


if __name__ == "__main__":
    unittest.main()
