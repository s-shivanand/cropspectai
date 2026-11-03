"""Exercise real sample predictions without serial access or saved output."""

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from streamlit.testing.v1 import AppTest

from app.sample_data import SAMPLE_FILE, SAMPLES_DIR, load_samples, predict_sample
from scripts import measure_predict as workflow


class SampleDataTests(unittest.TestCase):
    def test_samples_use_the_existing_model_and_feature_order(self):
        samples = load_samples()
        self.assertEqual(list(samples.columns), ["timestamp", "specimen", *workflow.CHANNELS])
        self.assertEqual(len(samples), 3)
        self.assertTrue(samples["specimen"].is_unique)
        self.assertTrue(all((SAMPLES_DIR / f"{label}.jpg").is_file() for label in samples["specimen"]))
        original = SAMPLE_FILE.read_bytes()
        with workflow.MODEL_CONFIG_PATH.open() as handle:
            features = json.load(handle)["features"]
        X = workflow.add_all_features(samples[workflow.CHANNELS])[features]
        expected = workflow.predict_model(X, workflow.MODEL_PATH)
        for (_, sample), cab in zip(samples.iterrows(), expected, strict=True):
            result = predict_sample(sample.to_dict(), sample["specimen"])
            self.assertAlmostEqual(result.cab, float(cab))
            self.assertEqual(result.measurement["timestamp"], sample["timestamp"])
        self.assertEqual(SAMPLE_FILE.read_bytes(), original)

    def test_choose_predict_switch_and_reset_without_hardware_or_saving(self):
        with tempfile.TemporaryDirectory() as directory, \
                patch.object(workflow, "CSV_FILE", Path(directory) / "measurements.csv"), \
                patch.object(workflow, "PREDICTION_FILE", Path(directory) / "predictions.csv"), \
                patch.object(workflow, "_check_device", side_effect=AssertionError("Device accessed")) as check, \
                patch.object(workflow.as7343_measure.serial, "Serial", side_effect=AssertionError("Serial opened")) as serial:
            app = AppTest.from_file(str(workflow.ml_config.PROJ_ROOT / "app/streamlit_app.py")).run()
            self.assertFalse(app.exception)
            self.assertEqual([button.label for button in app.button], ["Yes", "No"])
            self.assertFalse(app.text_input)
            app.button(key="device_no").click().run()
            self.assertFalse(app.exception)
            self.assertTrue(any("Don't have measuring device?" in text.value for text in app.markdown))
            self.assertFalse(app.dataframe)
            self.assertFalse(app.text_input)
            self.assertEqual(app.button[-1].key, "reset_app")
            app.button(key="reset_app").click().run()
            self.assertFalse(app.exception)
            self.assertEqual([button.label for button in app.button], ["Yes", "No"])
            app.button(key="device_no").click().run()
            app.button(key="use_sample_data").click().run()
            self.assertFalse(app.exception)
            self.assertEqual(len(app.dataframe[0].value), 3)
            self.assertEqual(app.button(key="reset_app").label, "Reset")
            self.assertEqual(app.button[-1].key, "reset_app")
            self.assertFalse(any(button.key == "toggle_live" for button in app.button))
            app.button(key="measure_predict").click().run()
            self.assertTrue(any("choose a sample" in warning.value for warning in app.warning))
            for label in load_samples()["specimen"]:
                app.selectbox(key="sample_specimen").select(label).run()
                photo_panel = next(block for block in app.get("flex_container")
                                   if block.proto.id.endswith("-sample_photo"))
                self.assertEqual(photo_panel.get("image")[0].proto.imgs[0].caption,
                                 label.replace("_", " "))
                self.assertFalse(app.text_input)
                self.assertTrue(any(f"Specimen chosen: {label}" in text.value for text in app.markdown))
                self.assertFalse(app.metric)
                app.button(key="measure_predict").click().run()
                self.assertFalse(app.exception)
                self.assertEqual(len(app.metric), 1)
                self.assertEqual(app.button(key="reset_app").label, "Reset")
                self.assertEqual(app.button[-1].key, "reset_app")
                self.assertEqual(app.session_state.measurement_result.measurement["specimen"], label)
                self.assertFalse(any(button.key == "track_specimen" for button in app.button))
            self.assertEqual(list(Path(directory).iterdir()), [])
            check.assert_not_called()
            serial.assert_not_called()
            app.button(key="device_yes").click().run()
            self.assertFalse(app.exception)
            self.assertEqual(app.button(key="measure_predict").label, "Measure and predict")
            self.assertEqual(app.button(key="reset_app").label, "Reset")
            self.assertEqual(app.button[-1].key, "reset_app")
            self.assertFalse(any(button.key == "use_sample_data" for button in app.button))
            self.assertFalse(app.button(key="toggle_live").disabled)
            self.assertEqual(app.text_input(key="specimen_label").value, "")
            self.assertFalse(app.metric)
            app.button(key="device_no").click().run()
            app.button(key="use_sample_data").click().run()
            app.selectbox(key="sample_specimen").select(load_samples().iloc[0]["specimen"]).run()
            app.button(key="reset_app").click().run()
            self.assertFalse(app.exception)
            self.assertFalse(app.text_input)
            self.assertFalse(app.selectbox)
            self.assertFalse(app.metric)
            self.assertEqual([button.label for button in app.button], ["Yes", "No"])
            self.assertNotIn("device_available", app.session_state)
            self.assertEqual(app.button(key="device_yes").proto.type, "secondary")
            self.assertEqual(app.button(key="device_no").proto.type, "secondary")
