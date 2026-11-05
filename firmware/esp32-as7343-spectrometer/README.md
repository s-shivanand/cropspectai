# ESP32–AS7343 measuring device

This PlatformIO project provides the firmware and Python acquisition helpers
used by [cropSpectAI](../../README.md). The ESP32 reads the AS7343 spectral sensor
and sends raw channel counts over USB serial. Calibration and Cab prediction run
in the Python backend, rather than on the ESP32.

[Device photo](data/device/device_setup.jpeg) ·
[Watch the demo on YouTube](https://youtu.be/MrCatPmRXBs) ·
[App and calibration guide](../../app/README.md)

## Build and upload

Use an ESP32 board compatible with the `esp32dev` configuration, an AS7343 sensor
connected through I²C, and a USB data cable. The repository’s configuration uses
the Arduino framework and Adafruit AS7343 library. Confirm the wiring for your
specific ESP32 and sensor board before powering the hardware.

Install [PlatformIO](https://docs.platformio.org/en/latest/core/installation/index.html).
If you use the VS Code extension, open **PlatformIO: New Terminal** from the
Command Palette so `pio` is available. A regular terminal needs PlatformIO on
its `PATH`.

From the repository root, build the firmware:

```sh
cd firmware/esp32-as7343-spectrometer
pio run -e esp32dev
```

Then, with the ESP32 connected, upload from the same folder:

```sh
pio run -e esp32dev -t upload
```

PlatformIO uses [`platformio.ini`](platformio.ini) to select the board and library.
If upload-port autodetection fails, configure the appropriate upload port for your
computer. Close the serial monitor before connecting from the app. For an already
compatible device, documentation or Python app changes do not require a reflash;
firmware changes require rebuilding and uploading.

## Measurement settings

[`src/main.cpp`](src/main.cpp) configures the sensor. Keep the metadata in
`SENSOR_SETTINGS` in [`src/as7343_measure.py`](src/as7343_measure.py) synchronized
when changing these values:

| Setting | Current value |
| --- | --- |
| Serial baud rate | 115200 |
| LED current | 20 mA |
| Gain | 64× |
| ATIME | 29 |
| ASTEP | 599 |
| Readings per batch | 10 |
| Minimum interval between batch reading starts | 1000 ms |
| Live reading interval | 500 ms |

[`include/channel_config.def`](include/channel_config.def) defines the shared
channel order, wavelengths, and display colours. The order is:
`F1, F2, FZ, F3, F4, FY, F5, FXL, F6, F7, F8, NIR`.
Python acquisition and the model must agree on channel names and order.

## Serial protocol

After initialization the firmware emits `AS7343_READY`. It accepts these
single-character commands:

| Command | Response |
| --- | --- |
| `l` | Enable live `DATA,<12 counts>` frames |
| `s` | Stop live streaming |
| `r` | Acquire ten `RAW,<12 counts>` frames followed by `RAW_DONE` |
| `m` | Acquire a batch and emit an `AVG` frame with raw means and sample deviations |

The app uses `l`, `s`, and `r`. Returning the raw batch lets Python normalize each
reading before averaging. Serial output also contains initialization and status
messages; the acquisition helpers distinguish those from data frames.

## Calibration and acquisition

[`data/ref/REF.csv`](data/ref/REF.csv) contains raw dark and white references.
These must match the device, sensor settings, and measurement conditions. See the
[reference format and hosted-app limitation](../../app/README.md#calibration).
A different serial-port name is fine: browser users select their own device in
the permission dialog. Python terminal users configure `SERIAL_PORT` in
`src/as7343_measure.py`.

To take measurements with Cab prediction, run the root project’s
[terminal workflow](../../docs/measurement_prediction.md). The
[`visualizer/`](visualizer/) folder contains Python spectrum visualization helpers;
[`notebooks/`](notebooks/) contains the initial measurement exploration.
