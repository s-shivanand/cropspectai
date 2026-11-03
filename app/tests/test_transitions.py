"""Check that reconnects preserve the spectrum and reset clears the retained frame."""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import streamlit as st
from streamlit.testing.v1 import AppTest

from app.live_sensor import LiveReading
from cropspectai import ml_config


class TransitionTests(unittest.TestCase):
    def test_first_reading_reconnect_and_reset(self):
        state = {"reading": LiveReading(), "owner": None}

        def start(owner):
            state["owner"] = owner
            state["reading"] = LiveReading(running=True)

        def stop(owner):
            state["reading"] = LiveReading(values=state["reading"].values)

        reader = SimpleNamespace(
            start=start, stop=stop,
            snapshot=lambda owner: state["reading"] if owner == state["owner"] else LiveReading(),
        )
        with patch("app.live_sensor.get_live_sensor", return_value=reader), \
                patch("streamlit.altair_chart", wraps=st.altair_chart) as render_plot:
            app = AppTest.from_file(str(ml_config.PROJ_ROOT / "app/streamlit_app.py")).run()
            self.assertFalse(app.get("vega_lite_chart"))
            app.button(key="toggle_live").click().run()
            self.assertFalse(app.exception)
            self.assertFalse(app.get("vega_lite_chart"))  # No invented readings during startup.
            state["reading"] = LiveReading(running=True, values=(0.2,) * 12)
            app.run()
            self.assertEqual(len(app.get("vega_lite_chart")), 1)
            self.assertEqual(render_plot.call_args.kwargs["key"], "spectrum_plot")
            panel = next(block for block in app.get("flex_container")
                         if block.proto.id.endswith("-spectrum_panel"))
            status = next(block for block in app.get("flex_container")
                          if block.proto.id.endswith("-spectrum_status"))
            self.assertFalse(any("Reading active" in item.value for item in panel.get("markdown")))
            self.assertTrue(any("Reading active" in item.value for item in status.get("markdown")))
            self.assertEqual(app.session_state.last_displayed_spectrum, (0.2,) * 12)
            app.button(key="toggle_live").click().run()
            app.button(key="toggle_live").click().run()
            self.assertFalse(app.exception)
            self.assertEqual(len(app.get("vega_lite_chart")), 1)
            self.assertEqual(len(app.get("image")), 1)
            self.assertEqual(render_plot.call_args.args[0].data["Reflectance"].tolist(), [0.2] * 12)
            self.assertTrue(any("Connecting · previous spectrum" in item.value for item in app.markdown))
            state["reading"] = LiveReading(running=True, values=(0.4,) * 12)
            app.run()
            self.assertEqual(render_plot.call_args.args[0].data["Reflectance"].tolist(), [0.4] * 12)
            app.button(key="reset_app").click().run()
            self.assertFalse(app.exception)
            self.assertFalse(app.get("vega_lite_chart"))
            self.assertNotIn("last_displayed_spectrum", app.session_state)


if __name__ == "__main__":
    unittest.main()
