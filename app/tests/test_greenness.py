"""Ensure the gauge uses the model feature without hiding invalid indices."""

import unittest
import base64
import json
import xml.etree.ElementTree as ET
from types import SimpleNamespace
from unittest.mock import patch

import streamlit as st
from app.tests.helpers import device_app

from app.greenness import greenness_gauge, spectrum_mndvi
from app.live_sensor import LiveReading
from cropspectai import ml_config


class GreennessTests(unittest.TestCase):
    def spectrum(self, nir=0.7, red=0.3, blue=0.1):
        channels = dict.fromkeys(ml_config.AS7343_CHANNELS, 0.2)
        channels.update(NIR=nir, F7=red, FZ=blue)
        return tuple(channels.values())

    def test_matches_existing_feature(self):
        self.assertAlmostEqual(spectrum_mndvi(self.spectrum()), 0.5)
        gauge = greenness_gauge(spectrum_mndvi(self.spectrum()))
        self.assertIn("MNDVI 0.500", gauge)
        self.assertIn("L 21 165.00", gauge)
        root = ET.fromstring(gauge)
        self.assertEqual(root.find("{http://www.w3.org/2000/svg}rect").attrib["y"], "25")
        ticks = root.findall("{http://www.w3.org/2000/svg}line")
        self.assertEqual(float(ticks[0].attrib["y1"]), 25)
        self.assertEqual(float(ticks[-1].attrib["y1"]), 305)

    def test_zero_denominator_has_no_marker(self):
        self.assertIsNone(spectrum_mndvi(self.spectrum(nir=0.3, red=0.3, blue=0.3)))
        gauge = greenness_gauge(None)
        self.assertIn("MNDVI N/A", gauge)
        self.assertIn("Index unavailable", gauge)
        self.assertNotIn("<path", gauge)

    def test_above_scale_keeps_actual_value(self):
        value = spectrum_mndvi(self.spectrum(nir=0.6, red=0.2, blue=0.35))
        self.assertAlmostEqual(value, 4)
        gauge = greenness_gauge(value)
        self.assertIn("MNDVI 4.000", gauge)
        self.assertIn("Above 0–1 scale", gauge)
        self.assertIn("L 21 25.00", gauge)

    def test_below_scale_keeps_actual_value(self):
        gauge = greenness_gauge(-0.5)
        self.assertIn("MNDVI -0.500", gauge)
        self.assertIn("Below 0–1 scale", gauge)
        self.assertIn("L 21 305.00", gauge)

    def test_indicator_tracks_the_displayed_spectrum(self):
        live = self.spectrum()
        averaged = self.spectrum(nir=0.6, red=0.2, blue=0.1)
        reader = SimpleNamespace(snapshot=lambda owner: LiveReading(values=live))
        with patch("app.live_sensor.get_live_sensor", return_value=reader), \
                patch("streamlit.image", wraps=st.image) as render_gauge:
            app = device_app(str(ml_config.PROJ_ROOT / "app/streamlit_app.py")).run()
            self.assertFalse(app.exception)
            self.assertIn("MNDVI 0.500", render_gauge.call_args.args[0])
            self.assertEqual(len(app.get("vega_lite_chart")), 1)
            image = app.get("image")[0].proto.imgs[0]
            self.assertTrue(image.url.startswith("data:image/svg+xml;base64,"))
            svg = base64.b64decode(image.url.split(",", 1)[1]).decode()
            self.assertIn('width="120" height="399"', svg)
            self.assertEqual(render_gauge.call_args.kwargs["width"], 120)
            rectangle = ET.fromstring(svg).find("{http://www.w3.org/2000/svg}rect")
            chart_spec = json.loads(app.get("vega_lite_chart")[0].proto.spec)
            self.assertEqual(float(rectangle.attrib["height"]), chart_spec["height"])
            self.assertEqual(chart_spec["autosize"], {"type": "fit-x", "contains": "content"})
            self.assertEqual(float(rectangle.attrib["width"]), 28)
            self.assertIn("1.00", svg)
            self.assertIn("0.00", svg)
            app.session_state.averaged_spectrum = averaged
            app.run()
            self.assertFalse(app.exception)
            self.assertIn("MNDVI 0.667", render_gauge.call_args.args[0])


if __name__ == "__main__":
    unittest.main()
