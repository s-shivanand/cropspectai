Run measurements with cab prediction
===================================

From the repository root, with the ESP32 connected and reference measurements
saved in the firmware's `data/ref/REF.csv`:

```sh
uv run python measure_predict.py
```

Enter a specimen label to measure, press Enter to reuse it, or enter `q` to quit.
Each successful measurement prints its predicted `cab` in µg/cm² and appends to:

- `firmware/esp32-as7343-spectrometer/data/misc/reflectance.csv`
- `data/prediction/misc/prediction.csv`

The prediction CSV contains `timestamp`, `specimen`, the twelve channel values
in sensor order, `noisy_channels`, and `cab`. The firmware CSV retains sensor
settings as usual. A prediction failure is printed and the measurement remains
saved in the firmware CSV.

Edit the settings section of `measure_predict.py` to configure a run:

```python
# Model
MODEL_PATH = config.MODELS_DIR / "xgb_v_01.joblib"
MODEL_CONFIG_PATH = config.MODELS_DIR / "xgb_best_parameters_v_01.json"

# Measurement and prediction data
CSV_FILE = channel_config.MISC_DATA_DIR / "reflectance.csv"
PREDICTION_FILE = config.PREDICTION_DATA_DIR / "misc" / "prediction.csv"
```

Keep the model and its JSON feature configuration paired. `CSV_FILE` controls
where measurements are appended, and `PREDICTION_FILE` controls where measurements
with predictions are appended. The orchestrator explicitly passes `CSV_FILE` to
the measurement script.
The destination's parent directory is created automatically, so a separate
`PATH` setting is unnecessary. Serial settings remain
in `firmware/esp32-as7343-spectrometer/src/as7343_measure.py`.
