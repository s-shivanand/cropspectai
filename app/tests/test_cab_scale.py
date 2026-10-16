"""Check the Cab reference interval, marker, and result-only rendering."""

import unittest
import xml.etree.ElementTree as ET

from streamlit.testing.v1 import AppTest

from app.cab_scale import TOOLSRTM_RANGE_URL, cab_scale
from cropspectai import config
from scripts.measure_predict import MeasurementResult


class CabScaleTests(unittest.TestCase):
    def test_marker_and_highlight(self):
        for value, expected_x in [(0, 24), (20, 110.4), (50, 240), (80, 369.6), (100, 456)]:
            with self.subTest(value=value):
                svg = cab_scale(value)
                ET.fromstring(svg)
                self.assertIn(f'L {expected_x:.2f} 37', svg)
                self.assertIn('x="110.4" y="12" width="259.2"', svg)

    def test_outside_scale_retains_value(self):
        self.assertIn('Predicted Cab 120.0000', cab_scale(120))
        self.assertIn('L 456.00 37', cab_scale(120))
        self.assertIn('outside the displayed', cab_scale(120))
        self.assertIn('Predicted Cab -5.0000', cab_scale(-5))
        self.assertIn('L 24.00 37', cab_scale(-5))

    def test_scale_appears_only_with_prediction(self):
        app = AppTest.from_file(str(config.PROJ_ROOT / 'app/streamlit_app.py')).run()
        self.assertFalse(app.get('image'))
        app.session_state.measurement_result = MeasurementResult(
            measurement={'specimen': 'leaf_01', 'timestamp': '2026-10-16T13:00:00'}, cab=42.5,
        )
        app.run()
        self.assertFalse(app.exception)
        self.assertEqual(app.metric[0].value, '42.5000 µg/cm²')
        self.assertEqual(len(app.get('image')), 1)
        self.assertTrue(any(TOOLSRTM_RANGE_URL in item.value for item in app.caption))


if __name__ == '__main__':
    unittest.main()
