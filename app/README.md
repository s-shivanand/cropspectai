# App guide

**[Open cropSpectAI →](https://cropspectai.streamlit.app/)** ·
[Demo video](https://youtu.be/MrCatPmRXBs) ·
[Local setup](../README.md#run-locally) · [Project overview](../README.md)

[Samples](#explore-without-hardware) · [Connect](#connect-a-measuring-device) ·
[Calibration](#using-your-own-device) · [Tracking](#track-a-specimen) ·
[Storage](#data-and-storage) · [Troubleshooting](#troubleshooting)

## Explore without hardware

1. Choose **No → Use sample data**.
2. Select a specimen to preview its photo, spectrum, and relative greenness.
3. Click **Predict Cab** to estimate chlorophyll content in µg/cm².

Repeat with any of the three specimens. Samples use the same model as device
measurements and do not save measurement, prediction, or tracking files.
The CSV and photos are in the [samples folder](../firmware/esp32-as7343-spectrometer/data/measurement/samples/).

## Connect a measuring device

**You need:** an ESP32–AS7343 device, a USB data cable, and desktop Chrome.

1. Connect the device and close other serial monitors.
2. Choose **Yes → Start**.
3. Select the ESP32’s USB serial port and click **Connect** in the browser dialog.
4. Position the leaf and wait for the live spectrum.
5. Enter a **Specimen label**, then click **Measure and predict**.
6. Keep the leaf still while ten readings are acquired. Review the average spectrum,
   predicted Cab, and any quality warnings.

| Control | Action |
| --- | --- |
| **Start / Stop** | Resume or pause live preview; preview frames are not saved |
| **Measure and predict** | Acquire a batch and save the measurement and prediction; preview need not be running first |
| **Reset** | Release the device and return to the unselected Yes/No question; saved files remain |

> [!TIP]
> Port names vary: look for `usbserial`, `CP210x`, `CH340`, or `USB Serial`.
> Unplug and reconnect the board to identify the entry that disappears and returns.
> Bluetooth pairing is unnecessary.

<details>
<summary>How the browser connection works</summary>

The browser reads the local device and sends the raw batch to the Python backend.
Calibration, feature engineering, prediction, and CSV writing happen on that backend.
The port stays connected across app reruns until Reset, disconnection, or page closure.

The app detects Web Serial support rather than the browser name. Use Chrome for
this documented workflow. Browser USB requires HTTPS or localhost. Browsers
without Web Serial can use samples; Python serial acquisition requires running
the server on the computer with the device. [Chrome Web Serial guide](https://developer.chrome.com/docs/capabilities/serial)

</details>

## Using your own device

Install the repository’s [firmware and settings](../firmware/esp32-as7343-spectrometer/README.md).
Compatible devices must match its channel order, baud rate, commands, and batch format.
The port name can differ: browser users select their own port without editing Python.

> [!IMPORTANT]
> The hosted app uses the checked-in calibration for **every device**. Matching
> firmware does not replace device-specific calibration. There is no calibration
> upload or selector; use your references in a local checkout or your own deployment.

### Calibration

The app reads raw dark/white counts from
[`data/ref/REF.csv`](../firmware/esp32-as7343-spectrometer/data/ref/REF.csv).

| Requirement | Value |
| --- | --- |
| Columns | `specimen` and the twelve configured channel names |
| Dark reference | One row labeled `BLACK_REF` or `DARK_REF` |
| White reference | One row labeled `WHITE_REF` |
| Values | Raw sensor counts, not normalized reflectance |

Each reading is normalized as `(raw - dark) / (white - dark)`, clamped to [0, 1],
then averaged across the batch. Channels with `white <= dark` produce zero and a warning.
References must match your device and measurement conditions. Recalibrate after
changing settings or measurement geometry.

## Track a specimen

After a successful device prediction, click **Track** to save the displayed result
to a specimen CSV. Further measurements with the same label append to that file
and the central prediction CSV while tracking remains enabled for the session.

| Control | Effect |
| --- | --- |
| **Track** | Save this prediction and enable specimen tracking |
| **Stop tracking** | Stop specimen-specific saves; retain existing records |
| **Continue to track** | Resume an existing file and include the displayed prediction |
| **Don't track** | Save only to the central CSV |
| **Reset** | Clear session choices; existing files remain |

Use a consistent label for the same leaf. The tracking view shows its saved history.
Labels are converted to safe filenames while their original text remains in the CSV.

## Data and storage

| App location | Where CSVs live |
| --- | --- |
| Hosted app | Streamlit server; temporary and shared across visitors |
| Local app | Your local project |

Hosted files may disappear on restart or redeployment, and identical labels can
combine visitors’ tracking records. There is no automatic GitHub commit, CSV
download, or permanent external storage. [Streamlit storage guidance](https://discuss.streamlit.io/t/storing-data-from-users/42346)

<details>
<summary>Output paths and save behavior</summary>

Paths are relative to the server’s project checkout:

| Output | Default path |
| --- | --- |
| Calibrated measurements and sensor metadata | `firmware/esp32-as7343-spectrometer/data/measurement/misc/reflectance.csv` |
| Features and Cab predictions | `data/prediction/misc/measure_prediction_v_01.csv` |
| Specimen history | `data/prediction/misc/tracked_specimens/<specimen>.csv` |

Directories and headers are created when needed. A prediction failure preserves
its successful measurement. Acquisition failures do not save incomplete batches;
ordinary reruns do not duplicate a browser measurement.

</details>

## Troubleshooting

| Problem | Try |
| --- | --- |
| No device in the port list | Use a USB data cable and check that the OS detects the serial adapter |
| Permission cancelled | Click **Start** or **Measure and predict** again |
| Port cannot open | Close other serial monitors, app tabs, and scripts using the port |
| Preview or batch times out | Check wiring and firmware, then **Reset** and reconnect |
| Implausible values | Check calibration, sensor settings, leaf placement, and quality warnings |
| No browser USB support | Use desktop Chrome, or try samples |
| History disappeared | A hosted restart or redeployment may have removed temporary files |

## Run locally

Follow the [installation steps](../README.md#run-locally), then run from the repository root:

```sh
uv run --locked streamlit run app/streamlit_app.py
```

For terminal acquisition or the Python entry point, see the
[measurement guide](../docs/measurement_prediction.md).
