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
