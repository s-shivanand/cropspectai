# CropSpectAI

AS7343 spectral measurement, feature engineering, and machine learning for
predicting leaf chlorophyll content (`cab`).

## Measure and predict

With the ESP32 connected and reference measurements saved, run from the
repository root:

```sh
uv run python measure_predict.py
```

Enter a specimen label to measure, press Enter to reuse the label, or enter `q`
to quit. Measurements are saved in the firmware data directory, and measurements
with predicted `cab` values are appended to `data/prediction/misc/prediction.csv`.

See [measurement and prediction instructions](docs/measurement_prediction.md)
for reference, model, and serial settings, and the
[firmware documentation](firmware/esp32-as7343-spectrometer/README.md) for the
sensor firmware.
