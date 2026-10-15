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
        app = AppTest.from_file(str(workflow.config.PROJ_ROOT / "app/streamlit_app.py"))
        app.run()
        self.assertFalse(app.exception)
        self.assertFalse(self.connections)
        app.text_input[0].set_value("leaf_ui")
        app.button[0].click().run()
        self.assertFalse(app.exception)
        self.assertEqual(len(app.metric), 1)
        self.assertIn("µg/cm²", app.metric[0].value)
        self.assertEqual(self.rows(self.measurements)[0]["specimen"], "leaf_ui")
        self.assertEqual(len(self.rows(self.predictions)), 1)
        app.run()
        self.assertEqual(len(app.metric), 1)
        self.assertEqual(len(self.connections), 1)
        # A new failed attempt must not display the preceding successful prediction.
        with patch.object(workflow.list_ports, "comports", return_value=[]):
            app.button[0].click().run()
        self.assertFalse(app.exception)
        self.assertEqual(len(app.metric), 0)
        self.assertIn("not connected", app.error[0].value)

    def test_app_validation_and_prediction_error(self):
        app = AppTest.from_file(str(workflow.config.PROJ_ROOT / "app/streamlit_app.py"))
        app.run()
        app.button[0].click().run()
        self.assertIn("specimen label", app.warning[0].value)
        self.assertFalse(self.connections)
        app.text_input[0].set_value("leaf_ui")
        with patch.object(workflow, "predict_model", side_effect=ValueError("model failed")):
            app.button[0].click().run()
        self.assertFalse(app.exception)
        self.assertEqual(len(app.metric), 0)
        self.assertIn("spectral measurement is saved", app.error[0].value)
        self.assertEqual(len(self.rows(self.measurements)), 1)


if __name__ == "__main__":
    unittest.main()
