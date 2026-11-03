Run measurements with cab prediction
===================================

From the repository root, with the ESP32 connected and reference measurements
saved in the firmware's `data/ref/REF.csv`:

```sh
uv run python scripts/measure_predict.py
```

Enter a specimen label to measure, press Enter to reuse it, or enter `q` to quit.
Each successful measurement prints its predicted `cab` in µg/cm² and appends to:

- `firmware/esp32-as7343-spectrometer/data/misc/reflectance.csv`
- `data/prediction/misc/measure_prediction.csv`

The prediction CSV contains `timestamp`, `specimen`, the twelve channel values
in sensor order, `noisy_channels`, all model features from the JSON configuration,
and `cab`. The firmware CSV retains sensor
settings as usual. A prediction failure is printed and the measurement remains
saved in the firmware CSV.

An interactive prediction CSV with the older header is upgraded automatically:
missing features are calculated from its saved channels and existing `cab` values
are preserved. No backup file is created.

If the configured serial port is unavailable, the script exits with
`esp32 device not connected` before changing any CSV files. Connect the ESP32
and verify `SERIAL_PORT` in the measurement script before trying again.

Edit the settings section of `scripts/measure_predict.py` to configure a run:

```python
# Model
MODEL_PATH = ml_config.MODELS_DIR / "xgb_v_01.joblib"
MODEL_CONFIG_PATH = ml_config.MODELS_DIR / "xgb_best_parameters_v_01.json"

# Measurement and prediction data
CSV_FILE = fimware_config.MISC_DATA_DIR / "reflectance.csv"
PREDICTION_FILE = ml_config.PREDICTION_DATA_DIR / "misc" / "measure_prediction.csv"
```

Keep the model and its JSON feature configuration paired. `CSV_FILE` controls
where measurements are appended, and `PREDICTION_FILE` controls where measurements
with predictions are appended. The orchestrator explicitly passes `CSV_FILE` to
the measurement script.
The destination's parent directory is created automatically, so a separate
`PATH` setting is unnecessary. Serial settings remain
in `firmware/esp32-as7343-spectrometer/src/as7343_measure.py`.

To predict an existing measurement CSV without connecting the sensor, edit the
four paths in `scripts/batch_predict.py` and run:

```sh
uv run python scripts/batch_predict.py
```

By default, this reads the firmware's `data/misc/reflectance.csv` and writes
`data/prediction/misc/batch_prediction.csv`, with the same output columns as the
interactive script. Each run replaces the batch output with predictions for all
input rows in their original order. The input CSV is left unchanged.
