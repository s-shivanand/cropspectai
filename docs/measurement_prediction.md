# Terminal measurement and prediction

[Project overview](../README.md) · [App guide](../app/README.md) ·
[Feature equations](feature_engineering.md)

Acquire measurements directly through Python and save Cab predictions to CSV.

## Before running

1. Complete the [local installation](../README.md#run-locally).
2. Install the [ESP32 firmware](../firmware/esp32-as7343-spectrometer/README.md).
3. Configure your device’s raw dark and white references in
   `firmware/esp32-as7343-spectrometer/data/ref/REF.csv`.
4. Set `SERIAL_PORT` in
   `firmware/esp32-as7343-spectrometer/src/as7343_measure.py` to the device’s port.
   Keep `BAUD_RATE` at `115200` to match the firmware. Close other serial monitors.

Reference format and normalization are described in the
[calibration guide](../app/README.md#calibration). The terminal workflow uses the
configured port; the browser workflow instead uses the browser’s port picker.

## Take measurements

From the repository root:

```sh
uv run --locked python scripts/measure_predict.py
```

Enter a specimen label to measure, press Enter to reuse it, or enter `q` to quit.
Keep the leaf still for ten readings. The workflow calibrates each reading,
calculates the mean spectrum, and predicts Cab using the configured model.

## Model and output settings

The settings section of [`scripts/measure_predict.py`](../scripts/measure_predict.py)
selects the model version and output paths. Current defaults are:

```python
MODEL_VERSION = "v_01"
MODEL_PATH = ml_config.MODELS_DIR / f"xgb_{MODEL_VERSION}.joblib"
MODEL_CONFIG_PATH = ml_config.MODELS_DIR / f"xgb_best_parameters_{MODEL_VERSION}.json"

CSV_FILE = fimware_config.MISC_DATA_DIR / "reflectance.csv"
PREDICTION_FILE = (
    ml_config.PREDICTION_DATA_DIR / "misc" / f"measure_prediction_{MODEL_VERSION}.csv"
)
```

Keep the model and its JSON feature configuration paired. The JSON determines
the feature selection and order used for inference. Changing output paths does
not change the acquisition or model logic.

| Output | Default path relative to repository root |
| --- | --- |
| Calibrated measurement | `firmware/esp32-as7343-spectrometer/data/measurement/misc/reflectance.csv` |
| Features and Cab prediction | `data/prediction/misc/measure_prediction_v_01.csv` |

The measurement CSV includes timestamp, specimen label, twelve reflectance
channels, noisy-channel descriptions, and sensor settings. The prediction CSV
includes the measurement metadata, channels, configured model features, and
`cab` in µg/cm². Parent directories and headers are created automatically.

| Outcome | Saved result |
| --- | --- |
| Measurement and prediction succeed | Measurement and prediction rows |
| Prediction fails | Measurement only, with the error reported |
| Acquisition fails | No incomplete measurement |
| Configured port is unavailable | Error reported before output CSVs are opened |

<details>
<summary><strong>Existing CSV compatibility</strong></summary>

The current prediction schema includes engineered features. When the workflow
encounters the supported older schema containing measurement columns and `cab`
only, it calculates the missing features and rewrites the file while preserving
existing Cab values. This migration does not create a backup. Other mismatched
schemas raise an error; use a new output path or move the incompatible file.

</details>

## Python entry point

```python
from scripts.measure_predict import measure_and_predict

result = measure_and_predict("leaf_01")
if result.cab is None:
    print(result.prediction_error)
else:
    print(f"Cab: {result.cab:.1f} µg/cm²")
```

The function acquires a batch, closes the serial connection, and saves to the
configured CSVs. Acquisition and file errors raise exceptions. Prediction errors
are returned in `MeasurementResult` so a successful measurement remains available.
Specimen tracking is an additional app action; this terminal entry point does
not independently enable tracking.
