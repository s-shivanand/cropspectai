/**
 * @file main.cpp
 * @brief Acquire AS7343 raw counts and serve measurement commands over serial.
 *
 * Commands: m sends batch means and sample standard deviations; r sends raw
 * readings followed by RAW_DONE; l enables live DATA frames; s stops streaming.
 * Channel order comes from channel_config.def. Keep sensor settings synchronized
 * with SENSOR_SETTINGS in as7343_measure.py.
 */
#include <Adafruit_AS7343.h>
#include <math.h>

// Expand the shared configuration into the library's channel indices.
#define CHANNEL(name, wavelength, color) AS7343_CHANNEL_##name,
const as7343_channel_t CHANNEL_INDICES[] = {
#include "channel_config.def"
};
#undef CHANNEL
const int N_CH = sizeof(CHANNEL_INDICES) / sizeof(CHANNEL_INDICES[0]);

Adafruit_AS7343 as7343;

// ============================================================
// SETTINGS
// ============================================================
const int N_AVG = 10;                        // Readings averaged per measurement
const unsigned long AVG_INTERVAL_MS = 1000;  // minimum interval between averaged reading starts
const unsigned long LIVE_INTERVAL_MS = 500;  // delay between live readings

bool liveMode = false;  // false = idle, true = stream DATA lines
unsigned long lastLiveMs = 0;

/**
 * @brief Read raw counts in the shared channel order.
 * @param vals Output buffer with space for N_CH values.
 * @return True when all channels were read; false on a sensor read failure.
 */
bool readChannels(float vals[N_CH]) {
  uint16_t r[18];
  if (!as7343.readAllChannels(r)) {
    return false;
  }
  for (int c = 0; c < N_CH; c++) {
    vals[c] = r[CHANNEL_INDICES[c]];
  }
  return true;
}

/**
 * @brief Emit one DATA frame with N_CH raw channel counts.
 *
 * Prints an error instead of a DATA frame if the sensor read fails.
 */
void sendLiveReading() {
  float v[N_CH];
  if (!readChannels(v)) {
    Serial.println("Read failed!");
    return;
  }
  Serial.print("DATA");
  for (int c = 0; c < N_CH; c++) {
    Serial.print(",");
    Serial.print((uint16_t)v[c]);
  }
  Serial.println();
}

/**
 * @brief Acquire N_AVG readings and emit raw frames or batch statistics.
 * @param sendRaw Emit RAW frames followed by RAW_DONE when true. Otherwise,
 *        emit an AVG frame containing N_CH means and N_CH sample deviations.
 *
 * Reading starts are separated by at least AVG_INTERVAL_MS. A sensor read
 * failure prints an error and aborts the batch without a completion frame.
 */
void takeAveragedMeasurement(bool sendRaw = false) {
  double sum[N_CH] = {0};
  double sumSq[N_CH] = {0};
  float v[N_CH];

  Serial.println("MEASURING (RAW COUNT)");

  for (int i = 0; i < N_AVG; i++) {
    unsigned long readingStartMs = millis();
    if (!readChannels(v)) {
      Serial.println("Read failed! Measurement aborted.");
      return;
    }
    if (sendRaw) {
      Serial.print("RAW");
      for (int c = 0; c < N_CH; c++) {
        Serial.print(",");
        Serial.print((uint16_t)v[c]);
      }
      Serial.println();
    }
    for (int c = 0; c < N_CH; c++) {
      sum[c] += v[c];
      sumSq[c] += (double)v[c] * v[c];
    }
    // Include acquisition time in the interval; do not wait after the last read.
    if (i < N_AVG - 1) {
      unsigned long elapsedMs = millis() - readingStartMs;
      if (elapsedMs < AVG_INTERVAL_MS) {
        delay(AVG_INTERVAL_MS - elapsedMs);
      }
    }
  }

  if (sendRaw) {
    Serial.println("RAW_DONE");
    return;
  }

  // Means
  Serial.print("AVG");
  for (int c = 0; c < N_CH; c++) {
    Serial.print(",");
    Serial.print(sum[c] / N_AVG, 2);
  }

  // Standard deviations (sample std dev, N-1)
  for (int c = 0; c < N_CH; c++) {
    double mean = sum[c] / N_AVG;
    double var = (sumSq[c] - N_AVG * mean * mean) / (N_AVG - 1);
    if (var < 0) var = 0;  // guard against tiny floating-point negatives
    Serial.print(",");
    Serial.print(sqrt(var), 2);
  }
  Serial.println();
}

/**
 * @brief Initialize serial, the sensor, and its LED before announcing readiness.
 *
 * Waits for the serial interface and halts if the sensor cannot be initialized.
 */
void setup() {
  Serial.begin(115200);
  while (!Serial) {
    delay(10);
  }

  Serial.println("Adafruit AS7343 Spectral Measurement");

  if (!as7343.begin()) {
    Serial.println("Could not find AS7343 sensor!");
    while (1) {
      delay(10);
    }
  }
  Serial.println("AS7343 found!");

  as7343.setLEDCurrent(20);  // 20 mA
  as7343.enableLED(true);    // LED stays on during the session

  // Configure sensor
  as7343.setGain(AS7343_GAIN_64X);
  as7343.setATIME(29);
  as7343.setASTEP(599);

  Serial.print("Integration time: ");
  Serial.print(as7343.getIntegrationTime());
  Serial.println(" ms");

  // Let the LED warm up and stabilize before any measurement
  delay(3000);

  Serial.println(
      "Commands: m = raw average, r = 10 raw readings for Python normalization, l = live stream ON, s = live stream OFF");
  Serial.println("AS7343_READY");
}

/**
 * @brief Process pending commands and send live readings when streaming is on.
 *
 * Measurement commands stop live streaming before starting their batch.
 */
void loop() {
  // 1. Handle commands from the computer
  while (Serial.available()) {
    char cmd = Serial.read();
    if (cmd == '\n' || cmd == '\r' || cmd == ' ') continue;

    if (cmd == 'm') {
      liveMode = false;  // Stop streaming before measuring
      takeAveragedMeasurement();
    } else if (cmd == 'r') {
      liveMode = false;
      takeAveragedMeasurement(true);
    } else if (cmd == 'l') {
      liveMode = true;
      Serial.println("LIVE_ON");
    } else if (cmd == 's') {
      liveMode = false;
      Serial.println("LIVE_OFF");
    }
  }

  // 2. Live streaming (only if switched on)
  if (liveMode && (millis() - lastLiveMs >= LIVE_INTERVAL_MS)) {
    lastLiveMs = millis();
    sendLiveReading();
  }
}
