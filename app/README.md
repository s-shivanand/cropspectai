# App guide

**[Open cropSpectAI →](https://cropspectai.streamlit.app/)** ·
[Demo video](https://youtu.be/MrCatPmRXBs) ·
[Local setup](../README.md#run-locally) · [Project overview](../README.md)

[Samples](#explore-without-hardware) · [Browsers](#browser-support) ·
[Connect](#connect-a-measuring-device) ·
[Calibration](#using-your-own-device) · [Tracking](#track-a-specimen) ·
[Storage](#data-and-storage) · [Troubleshooting](#troubleshooting)

## Explore without hardware

Use a modern desktop or mobile browser: Chrome, Edge, Firefox, Safari, Brave,
or Opera. Sample predictions do not need Web Serial or USB access.

1. Choose **No → Use sample data**.
2. Select a specimen to preview its photo, spectrum, and relative greenness.
3. Click **Predict Cab** to estimate chlorophyll content in µg/cm².

Repeat with any of the three specimens. Samples use the same model as device
measurements and do not save measurement, prediction, or tracking files.
The CSV and photos are in the [samples folder](../firmware/esp32-as7343-spectrometer/data/measurement/samples/).

## Browser support

USB acquisition requires a browser that exposes **Web Serial**, permission to
access the device, and HTTPS or localhost. Current desktop options are:

| Browser | Device connection |
| --- | --- |
| Chrome | Native Web Serial |
| Microsoft Edge | Native Web Serial |
| Opera | Native Web Serial |
| Firefox 151 or later | Web Serial; follow any additional permission/add-on prompt |
| Brave | Web Serial must be enabled; some versions require the Web Serial flag in `brave://flags` and a browser restart |
| Safari | No native Web Serial; use the sample workflow |

API availability does not guarantee that every browser/device combination has
been tested with this app. USB acquisition has been confirmed on macOS Chrome.
Phone USB support depends on the operating system and device; the current app
has no WebUSB fallback. Samples remain available on mobile browsers.

Sources: [browser compatibility](https://developer.mozilla.org/en-US/docs/Web/API/Serial#browser_compatibility),
[Firefox permissions](https://hacks.mozilla.org/2026/05/web-serial-support-in-firefox/),
and [Brave’s Web Serial option](https://github.com/brave/brave-browser/issues/24404).

## Connect a measuring device

**You need:** an ESP32–AS7343 device, a USB data cable, and a
[Web Serial-capable desktop browser](#browser-support).

1. Connect the device and close other serial monitors.
2. Choose **Yes → Start**.
3. [Identify the ESP32’s entry](#identify-the-esp32-port) in the serial-port picker,
   select it, and click **Connect**.
4. Position the leaf and wait for the live spectrum.
5. Enter a **Specimen label**, then click **Measure and predict**.
6. Keep the leaf still while ten readings are acquired. Review the average spectrum,
   predicted Cab, and any quality warnings.

| Control | Action |
| --- | --- |
| **Start / Stop** | Resume or pause live preview; preview frames are not saved |
| **Measure and predict** | Acquire a batch and save the measurement and prediction; preview need not be running first |
| **Reset** | Release the device and return to the unselected Yes/No question; saved files remain |

### Identify the ESP32 port

The picker may list several unrelated serial devices. Your board can appear as
**CP210x**, **CH340**, **USB Serial**, or a name containing **usbserial** or
**usbmodem**. Depending on the OS, it may also show a port such as `COM3`,
`/dev/ttyUSB0`, or `/dev/cu.usbserial-…`. These are examples, not fixed names.

If you cannot tell which entry is yours:

1. Cancel the picker and unplug the ESP32.
2. Click **Start** again and note the available entries, then cancel.
3. Reconnect the ESP32 using the USB data cable and open the picker again.
4. Select the new entry that appeared and click **Connect**.

There is no need to pair a Bluetooth device, select the author’s port name, or
edit `SERIAL_PORT` for browser acquisition. If no new entry appears, check the
USB cable/adapter, operating-system device detection, and browser support.

<details>
<summary>How the browser connection works</summary>

The browser reads the local device and sends the raw batch to the Python backend.
Calibration, feature engineering, prediction, and CSV writing happen on that backend.
The port stays connected across app reruns until Reset, disconnection, or page closure.

The app detects Web Serial support rather than the browser name. Browsers
without it can use samples; Python serial acquisition requires running the
server on the computer with the device. [Web Serial guide](https://developer.chrome.com/docs/capabilities/serial)

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
| No browser USB support | Use a [Web Serial-capable desktop browser](#browser-support), or try samples |
| History disappeared | A hosted restart or redeployment may have removed temporary files |

## Run locally

Follow the [installation steps](../README.md#run-locally), then run from the repository root:

```sh
uv run --locked streamlit run app/streamlit_app.py
```

For terminal acquisition or the Python entry point, see the
[measurement guide](../docs/measurement_prediction.md).
