"""Check browser calibration, prediction parity, and the unchanged app controls."""

import csv
import tempfile
import unittest
from contextlib import ExitStack
from pathlib import Path
from unittest.mock import patch

from app import browser_sensor as browser
from app.sample_data import predict_sample
from app.tests.helpers import device_app
from scripts import measure_predict as workflow


class BrowserSensorTests(unittest.TestCase):
    """Exercise browser batches without accessing hardware or repository outputs."""

    def setUp(self) -> None:
        """Keep all generated files in an isolated temporary directory."""
        self.stack = ExitStack()
        self.addCleanup(self.stack.close)
        self.directory = Path(self.stack.enter_context(tempfile.TemporaryDirectory()))
        self.measurements = self.directory / "measurements.csv"
        self.predictions = self.directory / "predictions.csv"
        self.stack.enter_context(patch.object(workflow, "CSV_FILE", self.measurements))
        self.stack.enter_context(patch.object(workflow, "PREDICTION_FILE", self.predictions))
        self.serial = self.stack.enter_context(
            patch.object(
                workflow.as7343_measure.serial,
                "Serial",
                side_effect=AssertionError("USB accessed on server"),
            )
        )
        self.references = workflow.as7343_measure.load_references(workflow.as7343_measure.REF_FILE)
        dark, white = self.references
        self.raw = [[(low + high) / 2 for low, high in zip(dark, white)]] * 10

    def rows(self, path: Path) -> list[dict[str, str]]:
        """Read generated CSV rows for assertions.

        Args:
            path: Temporary CSV output path.

        Returns:
            Saved rows with their header names.
        """
        with path.open(newline="") as handle:
            return list(csv.DictReader(handle))

    def test_same_normalization_and_model_as_local_workflow(self) -> None:
        """Match calibration, rounding, features, prediction, and output schemas."""
        sensor = workflow.as7343_measure
        # Include values outside reference bounds to test clamp-before-average.
        raw = [list(row) for row in self.raw]
        raw[0][0] = 0
        raw[1][0] = 65535
        with (
            patch.object(sensor, "DARK_REF", self.references[0], create=True),
            patch.object(sensor, "WHITE_REF", self.references[1], create=True),
        ):
            expected, _ = sensor.normalized_statistics(raw)
        result = browser.save_browser_measurement(raw, " leaf_01 ")
        self.assertEqual(result.measurement["specimen"], "leaf_01")
        self.assertEqual(
            [result.measurement[name] for name in workflow.CHANNELS],
            [round(value, 6) for value in expected],
        )
        self.assertEqual(result.cab, predict_sample(result.measurement, "leaf_01").cab)
        self.assertEqual(list(self.rows(self.measurements)[0]), sensor.HEADER)
        self.assertEqual(len(self.rows(self.predictions)), 1)
        self.assertTrue(any("saturation" in warning for warning in result.warnings))
        self.serial.assert_not_called()

    def test_bad_batches_never_create_output(self) -> None:
        """Reject wrong dimensions, nonnumeric values, and out-of-range counts."""
        for raw in (
            self.raw[:9],
            self.raw + [self.raw[0]],
            [[0] * 11] * 10,
            [[float("nan")] * 12] * 10,
            [[-1] * 12] * 10,
            [[65536] * 12] * 10,
            [[True] * 12] * 10,
            [["5"] * 12] * 10,
        ):
            with self.subTest(raw=raw), self.assertRaises(ValueError):
                browser.save_browser_measurement(raw, "leaf")
            self.assertEqual(list(self.directory.iterdir()), [])
        with self.assertRaises(ValueError):
            browser.save_browser_measurement(self.raw, " ")
        self.assertEqual(list(self.directory.iterdir()), [])

    def test_preview_does_not_mutate_firmware_globals(self) -> None:
        """Preview calibration is independent of local reader module globals."""
        sensor = workflow.as7343_measure
        before = {key: sensor.__dict__.get(key) for key in ("DARK_REF", "WHITE_REF", "ser")}
        reading = browser.browser_reading({"live": True, "values": self.raw[0]})
        self.assertEqual(reading.values, (0.5,) * 12)
        self.assertTrue(reading.running)
        self.assertEqual(before, {key: sensor.__dict__.get(key) for key in before})
        self.assertEqual(list(self.directory.iterdir()), [])

    def test_preview_preserves_saturation_warning(self) -> None:
        """Use the local preview's existing saturation warning text."""
        reading = browser.browser_reading({"live": True, "values": [65535] * 12})
        self.assertIn(
            "A channel is near saturation. Lower the gain or LED current.", reading.warnings
        )

    def test_prediction_failure_keeps_measurement(self) -> None:
        """Retain the local workflow's behavior when model inference fails."""
        with patch.object(workflow, "predict_model", side_effect=ValueError("model failed")):
            result = browser.save_browser_measurement(self.raw, "leaf")
        self.assertIsNone(result.cab)
        self.assertEqual(result.prediction_error, "model failed")
        self.assertEqual(len(self.rows(self.measurements)), 1)
        self.assertEqual(self.rows(self.predictions), [])

    def test_invalid_existing_header_is_not_modified(self) -> None:
        """Refuse to append to an incompatible measurement file."""
        self.measurements.write_text("wrong,header\n")
        with self.assertRaises(ValueError):
            browser.save_browser_measurement(self.raw, "leaf")
        self.assertEqual(self.measurements.read_text(), "wrong,header\n")
        self.assertFalse(self.predictions.exists())

    def test_browser_result_tracking_reruns_and_reset(self) -> None:
        """Keep the device controls and save each browser request only once."""
        packet = {"supported": True, "live": False, "busy": False}
        with patch.object(browser, "browser_sensor", side_effect=lambda session: packet.copy()):
            app = device_app(workflow.ml_config.PROJ_ROOT / "app/streamlit_app.py").run()
            self.assertFalse(app.exception)
            self.assertEqual(
                [button.label for button in app.button],
                ["Yes", "No", "Start", "Measure and predict", "Reset"],
            )
            packet.update(live=True, values=self.raw[0])
            app.run()
            self.assertFalse(app.exception)
            self.assertEqual(app.button(key="toggle_live").label, "Stop")
            self.assertEqual(len(app.get("vega_lite_chart")), 1)
            packet.update(
                live=False, request={"id": "one", "specimen": "browser_leaf"}, raw=self.raw
            )
            app.run()
            self.assertFalse(app.exception)
            self.assertEqual(len(app.metric), 1)
            app.run()
            self.assertFalse(app.exception)
            self.assertEqual(len(self.rows(self.measurements)), 1)
            self.assertEqual(len(self.rows(self.predictions)), 1)
            app.button(key="track_specimen").click().run()
            tracking = self.directory / "tracked_specimens" / "browser_leaf.csv"
            self.assertEqual(len(self.rows(tracking)), 1)
            packet.update(request={"id": "two", "specimen": "browser_leaf"})
            app.run()
            app.run()
            self.assertFalse(app.exception)
            self.assertEqual(len(self.rows(self.measurements)), 2)
            self.assertEqual(len(self.rows(self.predictions)), 2)
            self.assertEqual(len(self.rows(tracking)), 2)
            self.assertEqual(app.button[-1].key, "reset_app")
            app.button(key="reset_app").click().run()
            self.assertFalse(app.exception)
            self.assertEqual([button.label for button in app.button], ["Yes", "No"])
            self.assertNotIn("browser_request_id", app.session_state)
            self.serial.assert_not_called()

    def test_browser_errors_clear_stale_result_without_saving(self) -> None:
        """Cancellation and malformed batches surface through existing error slots."""
        packet = {
            "supported": True,
            "live": False,
            "busy": False,
            "request": {"id": "bad", "specimen": "leaf"},
            "raw": [[0] * 12],
        }
        with patch.object(browser, "browser_sensor", side_effect=lambda session: packet.copy()):
            app = device_app(workflow.ml_config.PROJ_ROOT / "app/streamlit_app.py").run()
            self.assertFalse(app.exception)
            self.assertTrue(app.error)
            self.assertFalse(app.metric)
            self.assertEqual(list(self.directory.iterdir()), [])
            packet.update(
                request={"id": "cancelled", "specimen": "leaf"},
                raw=None,
                error="No measuring device selected.",
            )
            app.run()
            self.assertFalse(app.exception)
            self.assertTrue(
                any("No measuring device selected" in error.value for error in app.error)
            )
            self.assertEqual(list(self.directory.iterdir()), [])
