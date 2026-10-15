# CropSpectAI app

From the repository root, run:

```sh
uv run streamlit run app/streamlit_app.py
```

Open the local URL printed by Streamlit. Connect the ESP32, position the leaf,
enter a specimen label, and press **Measure and predict**. Keep the leaf in place
until acquisition finishes. The app shows predicted Cab in µg/cm², the specimen
and timestamp, and any measurement warnings.

The app uses the serial port and saved references configured in
`firmware/esp32-as7343-spectrometer/src/as7343_measure.py`, and the model and CSV
paths configured in `scripts/measure_predict.py`. Each button press opens the
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
