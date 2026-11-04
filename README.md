# CropSpectAI

AS7343 spectral measurement, feature engineering, and machine learning for
predicting leaf chlorophyll content (`cab`).

## Measure and predict

With the ESP32 connected and reference measurements saved, run from the
repository root:

```sh
uv run python scripts/measure_predict.py
```

Enter a specimen label to measure, press Enter to reuse the label, or enter `q`
to quit. Measurements are saved under the firmware’s `data/measurement/` directory, and measurements
with predicted `cab` values are appended to `data/prediction/misc/measure_prediction.csv`.

To predict an existing measurement CSV without the sensor, configure the paths
in `scripts/batch_predict.py` and run `uv run python scripts/batch_predict.py`.

See [measurement and prediction instructions](docs/measurement_prediction.md)
for reference, model, and serial settings, and the
[firmware documentation](firmware/esp32-as7343-spectrometer/README.md) for the
sensor firmware.

## Streamlit app

Run `uv run streamlit run app/streamlit_app.py` locally, or open the deployed
app in desktop Chrome. Connect the ESP32 by USB, choose **Yes**, and press
**Start**. Select its serial port in Chrome's permission dialog. The existing
**Stop**, **Measure and predict**, tracking, and **Reset** controls work as before.
You can also measure directly without starting the preview first.

Close other serial monitors before connecting. Browser USB access requires HTTPS
(or localhost); no browser extension or firmware update is needed. Calibration
uses the repository's `firmware/esp32-as7343-spectrometer/data/ref/REF.csv`, which
must match the measuring device. Browsers without Web Serial keep the local
Python serial workflow; they cannot access a visitor's USB device from the
hosted app. The sample-data option works without USB.
