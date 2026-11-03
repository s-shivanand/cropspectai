"""Check streaming, port cleanup, and the preview-to-prediction handoff."""

import csv
import tempfile
import time
import unittest
from collections import deque
from contextlib import ExitStack
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import streamlit as st
from app.tests.helpers import device_app

from app.live_sensor import LiveSensor
from scripts import measure_predict as workflow


class StreamingSerial:
    def __init__(self, *args, **kwargs):
        self.lines = deque([b"AS7343_READY\n"])
        self.commands = []
        self.closed = False
        self.streaming = False

    def reset_input_buffer(self):
        self.lines.clear()

    def raw_line(self, prefix):
        sensor = workflow.as7343_measure
        fraction = 0.25 if prefix == "DATA" else 0.5
        raw = [dark + fraction * (white - dark)
               for dark, white in zip(sensor.DARK_REF, sensor.WHITE_REF)]
        return (prefix + "," + ",".join(str(value) for value in raw) + "\n").encode()

    def write(self, command):
        self.commands.append(command)
        self.streaming = command == b"l"
        if command == b"r":
            self.lines.extend([self.raw_line("RAW")] * 10)
            self.lines.append(b"RAW_DONE\n")

    def readline(self):
        if self.lines:
            return self.lines.popleft()
        if self.streaming:
            time.sleep(0.005)
            return self.raw_line("DATA")
        time.sleep(0.005)
        return b""

    def close(self):
        self.closed = True


class LiveSensorTests(unittest.TestCase):
    def setUp(self):
        stack = ExitStack()
        self.addCleanup(stack.close)
        self.directory = Path(stack.enter_context(tempfile.TemporaryDirectory()))
        self.measurements = self.directory / "reflectance.csv"
        self.predictions = self.directory / "predictions.csv"
        stack.enter_context(patch.object(workflow, "CSV_FILE", self.measurements))
        stack.enter_context(patch.object(workflow, "PREDICTION_FILE", self.predictions))
        stack.enter_context(patch.object(workflow.list_ports, "comports", return_value=[
            SimpleNamespace(device=workflow.as7343_measure.SERIAL_PORT)
        ]))
        self.connections = []

        def connect(*args, **kwargs):
            self.assertTrue(all(connection.closed for connection in self.connections))
            connection = StreamingSerial()
            self.connections.append(connection)
            return connection

        stack.enter_context(patch.object(workflow.as7343_measure.serial, "Serial", side_effect=connect))
        clock = SimpleNamespace(sleep=lambda _: None, monotonic=time.monotonic)
        stack.enter_context(patch.object(workflow, "time", clock))
        self.reader = LiveSensor()
        stack.enter_context(patch("app.live_sensor.get_live_sensor", return_value=self.reader))
        self.addCleanup(self.reader.close)

    def wait_for(self, predicate):
        deadline = time.monotonic() + 3
        while time.monotonic() < deadline:
            if predicate():
                return
            time.sleep(0.01)
        self.fail("Live reader did not reach the expected state")

    def test_stream_normalizes_without_saving_and_stop_releases_port(self):
        self.reader.start("session")
        self.wait_for(lambda: self.reader.snapshot("session").values is not None)
        reading = self.reader.snapshot("session")
        self.assertEqual(len(reading.values), 12)
        self.assertTrue(all(0 <= value <= 1 for value in reading.values))
        self.assertFalse(self.measurements.exists())
        self.assertFalse(self.predictions.exists())
        with self.assertRaises(RuntimeError):
            workflow.measure_and_predict("blocked")
        with self.assertRaises(RuntimeError):
            self.reader.start("other_session")
        self.reader.stop("other_session")
        self.assertTrue(self.reader.snapshot("session").running)
        self.reader.stop("session")
        self.assertTrue(self.connections[0].closed)
        self.assertEqual(self.connections[0].commands[-1], b"s")
        self.assertFalse(self.reader.snapshot("session").running)
        self.assertIsNotNone(workflow.measure_and_predict("after_preview").cab)

    def test_disconnect_and_malformed_frames(self):
        with patch.object(workflow.list_ports, "comports", return_value=[]):
            self.reader.start("session")
            self.wait_for(lambda: not self.reader.snapshot("session").running)
        self.assertIn("not connected", self.reader.snapshot("session").error)
        self.reader.start("session")
        self.wait_for(lambda: self.reader.snapshot("session").values is not None)
        self.connections[0].lines.extend([
            b"DATA,1,2\n", b"DATA,nan\n", b"DATA,not-a-number\n", b"Read failed\n",
        ])
        self.wait_for(lambda: not self.reader.snapshot("session").running)
        self.assertIn("could not read", self.reader.snapshot("session").error)
        self.assertTrue(self.connections[0].closed)

    def test_browser_disconnect_releases_port(self):
        self.reader.HEARTBEAT_TIMEOUT = 0.05
        self.reader.start("session")
        time.sleep(0.15)  # No snapshot heartbeats: simulate closing the browser.
        self.assertFalse(self.reader.snapshot("session").running)
        self.assertTrue(self.connections[0].closed)

    def test_reset_stops_stream_and_clears_preview(self):
        app = device_app(str(workflow.ml_config.PROJ_ROOT / "app/streamlit_app.py")).run()
        app.text_input[0].set_value("leaf_to_clear")
        app.button(key="toggle_live").click().run()
        owner = app.session_state.sensor_session
        self.wait_for(lambda: self.reader.snapshot(owner).values is not None)
        app.run()
        app.button(key="reset_app").click().run()
        self.assertFalse(app.exception)
        self.assertTrue(self.connections[0].closed)
        self.assertNotEqual(app.session_state.sensor_session, owner)
        self.assertFalse(app.text_input)
        self.assertEqual([button.label for button in app.button], ["Yes", "No"])
        self.assertEqual(len(app.get("vega_lite_chart")), 0)
        self.assertFalse(app.metric)
        self.assertFalse(app.error)
        self.assertFalse(self.measurements.exists())
        app.button(key="device_yes").click().run()
        app.button(key="toggle_live").click().run()
        self.wait_for(lambda: self.reader.snapshot(app.session_state.sensor_session).values is not None)
        self.assertEqual(len(self.connections), 2)

    def test_reset_clears_result_and_errors_without_changing_saved_data(self):
        app = device_app(str(workflow.ml_config.PROJ_ROOT / "app/streamlit_app.py")).run()
        app.text_input[0].set_value("saved_leaf")
        app.button(key="measure_predict").click().run()
        self.assertFalse(app.exception)
        self.assertEqual(len(app.metric), 1)
        saved_measurements = self.measurements.read_bytes()
        saved_predictions = self.predictions.read_bytes()
        app.session_state.measurement_error = "Previous measurement error"
        app.session_state.live_action_error = "Previous preview error"
        app.button(key="reset_app").click().run()
        self.assertFalse(app.exception)
        self.assertFalse(app.text_input)
        self.assertEqual([button.label for button in app.button], ["Yes", "No"])
        self.assertFalse(app.metric)
        self.assertFalse(app.error)
        self.assertFalse(app.warning)
        self.assertEqual(len(app.get("vega_lite_chart")), 0)
        self.assertEqual(self.measurements.read_bytes(), saved_measurements)
        self.assertEqual(self.predictions.read_bytes(), saved_predictions)
        app.run()
        self.assertEqual(len(app.get("vega_lite_chart")), 0)
        self.assertFalse(app.metric)

    def test_ui_start_chart_stop_and_predict(self):
        app = device_app(str(workflow.ml_config.PROJ_ROOT / "app/streamlit_app.py")).run()
        self.assertFalse(app.exception)
        app.button(key="toggle_live").click().run()
        owner = app.session_state.sensor_session
        self.wait_for(lambda: self.reader.snapshot(owner).values is not None)
        app.run()
        self.assertFalse(app.exception)
        self.assertEqual(app.button(key="toggle_live").label, "Stop")
        self.assertEqual(len(app.get("vega_lite_chart")), 1)
        self.assertEqual(len(app.subheader), 0)
        app.text_input[0].set_value("live_to_cab")
        preview_values = self.reader.snapshot(owner).values
        with patch("streamlit.altair_chart", wraps=st.altair_chart) as render_chart:
            app.button(key="measure_predict").click().run()
        plotted_average = render_chart.call_args.args[0].data["Reflectance"].tolist()
        plotted_data = render_chart.call_args.args[0].data
        self.assertFalse(app.exception)
        self.assertEqual(len(app.metric), 1)
        self.assertFalse(self.reader.snapshot(owner).running)
        self.assertEqual(app.button(key="toggle_live").label, "Start")
        self.assertNotEqual(tuple(plotted_average), preview_values)
        self.assertTrue(any("Average of 10 spectral readings" in item.value for item in app.markdown))
        self.assertEqual([item.value for item in app.subheader], ["Predicted Cab"])
        wavelengths = plotted_data["Wavelength (nm)"].astype(int).tolist()
        self.assertEqual(wavelengths, sorted(workflow.fimware_config.WAVELENGTHS))
        self.assertEqual(len(self.connections), 2)
        with self.measurements.open(newline="") as handle:
            rows = list(csv.DictReader(handle))
            self.assertEqual(len(rows), 1)
            self.assertEqual(plotted_average, [float(rows[0][name])
                                              for name in plotted_data["Channel"]])
            self.assertEqual(app.session_state.averaged_spectrum,
                             tuple(float(rows[0][name])
                                   for name in workflow.fimware_config.CHANNEL_NAMES))
        with self.predictions.open(newline="") as handle:
            self.assertEqual(len(list(csv.DictReader(handle))), 1)
        # Restarting the live preview replaces the measured spectrum again.
        app.button(key="toggle_live").click().run()
        self.assertNotIn("averaged_spectrum", app.session_state)
        self.wait_for(lambda: self.reader.snapshot(owner).values is not None)
        app.run()
        self.assertFalse(app.exception)
        self.assertTrue(any("Reading active" in item.value for item in app.markdown))
        self.assertEqual([item.value for item in app.subheader], ["Predicted Cab"])


if __name__ == "__main__":
    unittest.main()
