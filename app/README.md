# CropSpectAI app

From the repository root, run:

```sh
uv run streamlit run app/streamlit_app.py
```

Open the local URL printed by Streamlit. Connect the ESP32, position the leaf,
and press **Start reading** under **Reading sensor** to see live reflectance bars
for the twelve spectral channels. The chart uses the saved dark and white
references, with reflectance on a fixed 0–1 scale. Preview readings are not saved.

Enter a specimen label below the chart and press **Measure and predict**. The
preview stops automatically before the measurement begins. A message explains
that ten readings are being averaged; the chart then shows the averaged spectrum
used for prediction, including when prediction fails after a saved measurement.
Keep the leaf in place until acquisition finishes. The app shows
predicted Cab in µg/cm² and any measurement warnings. Press **Start reading** to
resume the preview, or **Stop reading** to stop it without taking a measurement.

The app uses the serial port and saved references configured in
`firmware/esp32-as7343-spectrometer/src/as7343_measure.py`, and the model and CSV
paths configured in `scripts/measure_predict.py`. Each **Measure and predict** press opens the
connection, acquires the existing ten-reading batch, saves the measurement and
prediction, and closes the connection. A prediction failure leaves the spectral
measurement saved. Ordinary app reruns do not acquire another measurement.

After a successful prediction, press **Track** below the Cab value to save that
prediction to `tracked_specimens/<specimen>.csv` alongside the usual prediction
CSV. The specimen CSV has exactly the same columns and values, including model
features. Future app measurements with the same specimen label append to both
files for the rest of the session. While enabled, **Stop tracking** replaces
**Track**. Stopping keeps saved records and returns future measurements to central
CSV saves only. Tracking can be resumed without duplicating the displayed result.
After Reset or an app restart, an existing specimen file prompts **Continue to
track** or **Don't track**. Until you continue, predictions are saved only in the
central CSV. Continuing also adds the currently displayed prediction to the
specimen file. Other specimen labels are not tracked until explicitly enabled.
Reset clears the screen and session choices without removing saved tracking
files. Labels containing filename separators are safely
encoded in the filename while retaining the original label inside the CSV.

The terminal workflow remains available:

```sh
uv run python scripts/measure_predict.py
```

The same script also exposes a Python entry point:

```python
from scripts.measure_predict import measure_and_predict

result = measure_and_predict(specimen="leaf_01")
print(result.cab)  # None if prediction failed; inspect result.prediction_error.
```

Run the hardware-free regression checks, which use temporary CSVs and the saved
model, from the repository root:

```sh
uv run python -m unittest discover -s app/tests -v
```
