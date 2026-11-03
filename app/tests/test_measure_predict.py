"""Hardware-free regression checks: python -m unittest discover -s app/tests."""

import csv
import io
import math
import tempfile
import time
import unittest
from collections import deque
from contextlib import ExitStack, redirect_stdout
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from streamlit.testing.v1 import AppTest

from app.tracking import specimen_tracking_path
from scripts import measure_predict as workflow


class SimulatedSerial:
    """Exercise the real RAW parser with one complete firmware batch per request."""

    def __init__(self, *args, **kwargs):
        self.lines = deque([b"AS7343_READY\n"])
        self.commands = []
        self.closed = False

    def reset_input_buffer(self):
        self.lines.clear()

    def write(self, command):
        self.commands.append(command)
        if command == b"r":
            sensor = workflow.as7343_measure
            values = [(dark + white) / 2 for dark, white in zip(
                sensor.DARK_REF, sensor.WHITE_REF
            )]
            line = ("RAW," + ",".join(str(value) for value in values) + "\n").encode()
            self.lines.extend([line] * sensor.SENSOR_SETTINGS["n_avg"])
            self.lines.append(b"RAW_DONE\n")

    def readline(self):
        return self.lines.popleft() if self.lines else b""

    def close(self):
        self.closed = True


class MeasurementTests(unittest.TestCase):
    def setUp(self):
        self.stack = ExitStack()
        self.addCleanup(self.stack.close)
        self.directory = Path(self.stack.enter_context(tempfile.TemporaryDirectory()))
        self.measurements = self.directory / "reflectance.csv"
        self.predictions = self.directory / "predictions.csv"
        self.connections = []
        self.stack.enter_context(patch.object(workflow, "CSV_FILE", self.measurements))
        self.stack.enter_context(patch.object(workflow, "PREDICTION_FILE", self.predictions))
        self.stack.enter_context(patch.object(workflow.list_ports, "comports", return_value=[
            SimpleNamespace(device=workflow.as7343_measure.SERIAL_PORT)
        ]))
        self.stack.enter_context(patch.object(workflow.as7343_measure.serial, "Serial",
                                             side_effect=self.connect))
        sensor_clock = SimpleNamespace(sleep=lambda _: None, monotonic=time.monotonic,
                                       time=time.time)
        self.stack.enter_context(patch.object(workflow, "time", sensor_clock))
        self.stack.enter_context(patch.object(workflow.as7343_measure, "time", sensor_clock))
        self.stack.enter_context(redirect_stdout(io.StringIO()))

    def connect(self, *args, **kwargs):
        connection = SimulatedSerial(*args, **kwargs)
        self.connections.append(connection)
        return connection

    def rows(self, path):
        with path.open(newline="") as handle:
            return list(csv.DictReader(handle))

    def test_one_batch_real_model_and_append(self):
        sensor = workflow.as7343_measure
        previous = {name: (hasattr(sensor, name), getattr(sensor, name, None))
                    for name in ("ser", "DARK_REF", "WHITE_REF")}
        result = workflow.measure_and_predict(" leaf_01 ")
        self.assertTrue(math.isfinite(result.cab))
        self.assertEqual(result.measurement["specimen"], "leaf_01")
        self.assertIsNone(result.prediction_error)
        self.assertEqual(self.connections[0].commands.count(b"r"), 1)
        self.assertTrue(self.connections[0].closed)
        self.assertEqual(previous, {name: (hasattr(sensor, name), getattr(sensor, name, None))
                                    for name in previous})
        workflow.measure_and_predict("leaf_02")
        self.assertEqual(len(self.rows(self.measurements)), 2)
        predictions = self.rows(self.predictions)
        self.assertEqual(len(predictions), 2)
        self.assertEqual(float(predictions[0]["cab"]), result.cab)

    def test_missing_device_and_empty_label_do_not_write(self):
        with self.assertRaises(ValueError):
            workflow.measure_and_predict(" ")
        with patch.object(workflow.list_ports, "comports", return_value=[]):
            with self.assertRaises(ConnectionError):
                workflow.measure_and_predict("leaf_01")
        self.assertFalse(self.measurements.exists())
        self.assertFalse(self.predictions.exists())
        self.assertFalse(self.connections)

    def test_saved_mndvi_is_never_clipped_to_the_display_scale(self):
        cases = [(0.6, 0.2, 0.35, 4.0), (0.2, 0.4, 0.1, -0.5),
                 (0.3, 0.3, 0.3, None)]
        with patch.object(workflow, "predict_model", return_value=[10.0]), \
                workflow._prediction_output() as (features, writer, handle):
            for nir, red, blue, expected in cases:
                measurement = dict.fromkeys(workflow.CHANNELS, 0.2)
                measurement.update(NIR=nir, F7=red, FZ=blue, timestamp="2026-10-16T12:00:00",
                                   specimen="mndvi_check", noisy_channels="")
                result = workflow._save_prediction(measurement, features, writer, handle)
                self.assertIsNotNone(result.cab)
        rows = self.rows(self.predictions)
        for row, (_, _, _, expected) in zip(rows, cases):
            if expected is None:
                self.assertTrue(math.isnan(float(row["mndvi_as7343"])))
            else:
                self.assertAlmostEqual(float(row["mndvi_as7343"]), expected)

    def test_prediction_failure_keeps_measurement(self):
        with patch.object(workflow, "predict_model", side_effect=ValueError("model failed")):
            result = workflow.measure_and_predict("leaf_01")
        self.assertIsNone(result.cab)
        self.assertEqual(result.prediction_error, "model failed")
        self.assertEqual(len(self.rows(self.measurements)), 1)
        self.assertEqual(self.rows(self.predictions), [])
        self.assertTrue(self.connections[0].closed)

    def test_failed_acquisition_closes_port_and_releases_lock(self):
        with patch.object(workflow.as7343_measure, "take_measurement", return_value=None):
            with self.assertRaises(RuntimeError):
                workflow.measure_and_predict("leaf_01")
        self.assertFalse(self.measurements.exists())
        self.assertEqual(self.rows(self.predictions), [])
        self.assertTrue(self.connections[0].closed)
        self.assertIsNotNone(workflow.measure_and_predict("retry").cab)

    def test_existing_prediction_header_is_upgraded(self):
        measurement = workflow.measure_and_predict("leaf_01").measurement
        old_columns = [*workflow.MEASUREMENT_COLUMNS, "cab"]
        with self.predictions.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=old_columns)
            writer.writeheader()
            writer.writerow({**{name: measurement[name] for name in workflow.MEASUREMENT_COLUMNS},
                             "cab": 12.34})
        workflow.measure_and_predict("leaf_02")
        rows = self.rows(self.predictions)
        self.assertEqual(len(rows), 2)
        self.assertEqual(float(rows[0]["cab"]), 12.34)
        self.assertGreater(len(rows[0]), len(old_columns))

    def test_terminal_label_reuse_and_quit_still_work(self):
        with patch("builtins.input", side_effect=["leaf_01", "", "q"]):
            workflow.main()
        self.assertEqual([row["specimen"] for row in self.rows(self.measurements)],
                         ["leaf_01", "leaf_01"])
        self.assertEqual(len(self.rows(self.predictions)), 2)
        self.assertEqual(len(self.connections), 1)
        self.assertEqual(self.connections[0].commands.count(b"r"), 2)
        self.assertTrue(self.connections[0].closed)

    def test_button_to_real_prediction_and_rerun(self):
        app = AppTest.from_file(str(workflow.ml_config.PROJ_ROOT / "app/streamlit_app.py"))
        app.run()
        self.assertFalse(app.exception)
        self.assertFalse(self.connections)
        app.text_input[0].set_value("leaf_ui")
        app.button(key="measure_predict").click().run()
        self.assertFalse(app.exception)
        self.assertEqual(len(app.metric), 1)
        self.assertEqual(len(app.get("vega_lite_chart")), 1)
        self.assertTrue(any("Average of 10 spectral readings" in item.value for item in app.markdown))
        self.assertTrue(any(item.value == "Predicted Cab" for item in app.subheader))
        self.assertTrue(any("Specimen:" in item.value and "leaf_ui" in item.value
                            and "Measured at" in item.value for item in app.markdown))
        self.assertIn("µg/cm²", app.metric[0].value)
        self.assertEqual(self.rows(self.measurements)[0]["specimen"], "leaf_ui")
        self.assertEqual(len(self.rows(self.predictions)), 1)
        # Only a new measurement asks the browser to reveal and scroll to the card.
        reveals = [frame for frame in app.get("iframe")
                   if "scrollIntoView" in frame.proto.srcdoc]
        self.assertEqual(len(reveals), 1)
        self.assertIn("card.animate", reveals[0].proto.srcdoc)
        self.assertIn("prefers-reduced-motion", reveals[0].proto.srcdoc)
        self.assertNotIn("reveal_result", app.session_state)
        app.run()
        self.assertFalse(app.get("iframe"))
        self.assertEqual(len(app.metric), 1)
        self.assertEqual(len(self.connections), 1)
        # A new failed attempt must not display the preceding successful prediction.
        with patch.object(workflow.list_ports, "comports", return_value=[]):
            app.button(key="measure_predict").click().run()
        self.assertFalse(app.exception)
        self.assertEqual(len(app.metric), 0)
        self.assertIn("not connected", app.error[0].value)

    def test_track_current_prediction_and_future_measurements_of_only_that_specimen(self):
        app_path = str(workflow.ml_config.PROJ_ROOT / "app/streamlit_app.py")
        app = AppTest.from_file(app_path).run()
        app.text_input[0].set_value("leaf_01")
        app.button(key="measure_predict").click().run()
        tracking_file = specimen_tracking_path("leaf_01", self.predictions)
        self.assertFalse(tracking_file.exists())
        self.assertFalse(app.button(key="track_specimen").disabled)
        app.button(key="track_specimen").click().run()
        self.assertFalse(app.exception)
        self.assertNotIn("track_specimen", [button.key for button in app.button])
        self.assertTrue(any("Tracking enabled · Specimen: leaf_01" in item.value for item in app.caption))
        self.assertFalse(any("leaf_01.csv" in item.value for item in app.caption))
        self.assertEqual(tracking_file.read_bytes(), self.predictions.read_bytes())
        app.run()
        self.assertEqual(len(self.rows(tracking_file)), 1)
        app.button(key="measure_predict").click().run()
        self.assertFalse(app.exception)
        self.assertEqual(len(self.rows(tracking_file)), 2)
        self.assertEqual(tracking_file.read_bytes(), self.predictions.read_bytes())
        self.assertNotIn("track_specimen", [button.key for button in app.button])
        app.text_input[0].set_value("other_leaf")
        app.button(key="measure_predict").click().run()
        self.assertFalse(app.exception)
        self.assertEqual(len(self.rows(tracking_file)), 2)
        self.assertFalse(specimen_tracking_path("other_leaf", self.predictions).exists())
        self.assertFalse(app.button(key="track_specimen").disabled)
        # A fresh app session asks before adding its current prediction to the specimen CSV.
        fresh_app = AppTest.from_file(app_path).run()
        fresh_app.text_input[0].set_value("leaf_01")
        fresh_app.button(key="measure_predict").click().run()
        self.assertFalse(fresh_app.exception)
        self.assertEqual(len(self.rows(tracking_file)), 2)
        self.assertTrue(any("Continue to track?" in item.value for item in fresh_app.caption))
        fresh_app.button(key="continue_tracking").click().run()
        self.assertFalse(fresh_app.exception)
        self.assertEqual(len(self.rows(tracking_file)), 3)
        self.assertEqual(len(self.rows(self.predictions)), 4)
        self.assertEqual(list(self.rows(tracking_file)[-1]), list(self.rows(self.predictions)[-1]))
        self.assertEqual(self.rows(tracking_file)[-1], self.rows(self.predictions)[-1])
        before_reset = tracking_file.read_bytes()
        fresh_app.button(key="reset_app").click().run()
        self.assertEqual(tracking_file.read_bytes(), before_reset)
        fresh_app.text_input[0].set_value("leaf_01")
        fresh_app.button(key="measure_predict").click().run()
        self.assertTrue(any("Continue to track?" in item.value for item in fresh_app.caption))
        self.assertEqual(tracking_file.read_bytes(), before_reset)
        fresh_app.button(key="decline_tracking").click().run()
        self.assertFalse(fresh_app.exception)
        self.assertEqual(tracking_file.read_bytes(), before_reset)
        fresh_app.button(key="measure_predict").click().run()
        self.assertFalse(fresh_app.exception)
        self.assertEqual(tracking_file.read_bytes(), before_reset)
        self.assertEqual(len(self.rows(self.predictions)), 6)
        self.assertNotIn("continue_tracking", [button.key for button in fresh_app.button])
        self.assertIn("track_specimen", [button.key for button in fresh_app.button])

    def test_stop_tracking_keeps_saved_rows_and_can_resume_without_duplicates(self):
        app = AppTest.from_file(str(workflow.ml_config.PROJ_ROOT / "app/streamlit_app.py")).run()
        app.text_input[0].set_value("leaf_01")
        app.button(key="measure_predict").click().run()
        self.assertEqual(len(app.get("vega_lite_chart")), 1)
        app.button(key="track_specimen").click().run()
        tracking_file = specimen_tracking_path("leaf_01", self.predictions)
        self.assertFalse(app.exception)
        self.assertEqual(len(app.get("vega_lite_chart")), 3)
        self.assertTrue(any(item.value == "Specimen progression" for item in app.subheader))
        self.assertIn("stop_tracking", [button.key for button in app.button])
        self.assertNotIn("track_specimen", [button.key for button in app.button])
        saved = tracking_file.read_bytes()
        app.button(key="stop_tracking").click().run()
        self.assertFalse(app.exception)
        self.assertIn("track_specimen", [button.key for button in app.button])
        self.assertNotIn("stop_tracking", [button.key for button in app.button])
        self.assertEqual(len(app.get("vega_lite_chart")), 1)
        self.assertFalse(any(item.value == "Specimen progression" for item in app.subheader))
        self.assertEqual(tracking_file.read_bytes(), saved)
        # Resuming the same displayed result must not append it twice.
        app.button(key="track_specimen").click().run()
        self.assertEqual(tracking_file.read_bytes(), saved)
        app.button(key="stop_tracking").click().run()
        app.button(key="measure_predict").click().run()
        self.assertFalse(app.exception)
        self.assertEqual(tracking_file.read_bytes(), saved)
        self.assertEqual(len(self.rows(self.predictions)), 2)
        app.button(key="track_specimen").click().run()
        self.assertEqual(len(self.rows(tracking_file)), 2)
        app.button(key="measure_predict").click().run()
        self.assertFalse(app.exception)
        self.assertEqual(len(self.rows(tracking_file)), 3)
        self.assertEqual(len(self.rows(self.predictions)), 3)
        self.assertIn("stop_tracking", [button.key for button in app.button])

    def test_track_write_error_preserves_prediction_and_can_be_retried(self):
        app = AppTest.from_file(str(workflow.ml_config.PROJ_ROOT / "app/streamlit_app.py")).run()
        app.text_input[0].set_value("leaf_01")
        app.button(key="measure_predict").click().run()
        original = self.predictions.read_bytes()
        with patch("app.tracking.append_tracking_prediction", side_effect=OSError("Disk unavailable")):
            app.button(key="track_specimen").click().run()
        self.assertFalse(app.exception)
        self.assertTrue(any("Disk unavailable" in item.value for item in app.error))
        self.assertEqual(self.predictions.read_bytes(), original)
        self.assertEqual(len(app.metric), 1)
        self.assertFalse(app.button(key="track_specimen").disabled)
        app.button(key="track_specimen").click().run()
        self.assertFalse(app.exception)
        self.assertFalse(app.error)
        tracking_file = specimen_tracking_path("leaf_01", self.predictions)
        self.assertEqual(tracking_file.read_bytes(), original)

    def test_app_validation_and_prediction_error(self):
        app = AppTest.from_file(str(workflow.ml_config.PROJ_ROOT / "app/streamlit_app.py"))
        app.run()
        app.button(key="measure_predict").click().run()
        self.assertIn("specimen label", app.warning[0].value)
        self.assertFalse(self.connections)
        app.text_input[0].set_value("leaf_ui")
        with patch.object(workflow, "predict_model", side_effect=ValueError("model failed")):
            app.button(key="measure_predict").click().run()
        self.assertFalse(app.exception)
        self.assertEqual(len(app.metric), 0)
        self.assertIn("spectral measurement is saved", app.error[0].value)
        self.assertEqual(len(app.get("vega_lite_chart")), 1)
        self.assertEqual(len(self.rows(self.measurements)), 1)


if __name__ == "__main__":
    unittest.main()
