"""Manage live firmware streaming independently of Streamlit page reruns."""

import atexit
import math
import time
from dataclasses import dataclass
from threading import Event, Lock, Thread

import streamlit as st

from scripts import measure_predict as workflow


@dataclass(frozen=True)
class LiveReading:
    running: bool = False
    values: tuple[float, ...] | None = None
    error: str | None = None
    warnings: tuple[str, ...] = ()


class LiveSensor:
    """One serial reader, protected from overlapping sessions and measurements."""

    HEARTBEAT_TIMEOUT = 20
    DATA_TIMEOUT = 15

    def __init__(self):
        self._lock = Lock()
        self._thread = None
        self._stop = Event()
        self._owner = None
        self._heartbeat = 0
        self._reading = LiveReading()

    def start(self, owner):
        with self._lock:
            if self._thread is not None and self._thread.is_alive():
                if self._owner != owner:
                    raise RuntimeError("The sensor is being used in another app session.")
                return
            self._owner = owner
            self._heartbeat = time.monotonic()
            self._reading = LiveReading(running=True)
            self._stop = Event()
            self._thread = Thread(target=self._read, name="spectral-preview", daemon=True)
            self._thread.start()

    def snapshot(self, owner):
        with self._lock:
            if owner != self._owner:
                return LiveReading()
            self._heartbeat = time.monotonic()
            return self._reading

    def stop(self, owner):
        with self._lock:
            if owner != self._owner:
                return
            self._stop.set()
            thread = self._thread
        # Includes the existing ESP32 boot wait, plus a bounded serial read.
        if thread is not None:
            thread.join(timeout=12)
            if thread.is_alive():
                raise RuntimeError("The live sensor is still stopping. Please try again shortly.")

    def close(self):
        self.stop(self._owner)

    def _read(self):
        warnings = []
        try:
            with workflow._exclusive_measurement():
                workflow._check_device()
                with workflow._sensor_connection(warnings):
                    sensor = workflow.as7343_measure
                    if self._stop.is_set():
                        return
                    sensor.ser.write(b"l")
                    last_data = time.monotonic()
                    while not self._stop.is_set():
                        with self._lock:
                            heartbeat = self._heartbeat
                        if time.monotonic() - heartbeat > self.HEARTBEAT_TIMEOUT:
                            break  # Release the port when the browser stops refreshing.
                        if time.monotonic() - last_data > self.DATA_TIMEOUT:
                            raise TimeoutError("No live readings received. Check the sensor and try again.")
                        line = sensor.ser.readline().decode("utf-8", errors="ignore").strip()
                        if line.startswith("Read failed"):
                            raise RuntimeError("The sensor could not read. Please try again.")
                        if not line.startswith("DATA,"):
                            continue
                        try:
                            raw = [float(value) for value in line.split(",")[1:]]
                        except ValueError:
                            continue
                        if len(raw) != sensor.N_CH or not all(
                            math.isfinite(value) and 0 <= value <= 65535 for value in raw
                        ):
                            continue
                        values = tuple(sensor.normalize(raw))
                        last_data = time.monotonic()
                        visible_warnings = [message for message in warnings
                                            if not message.startswith("ESP32 readiness")]
                        if max(raw) > 60000:
                            visible_warnings.append("A channel is near saturation. "
                                                    "Lower the gain or LED current.")
                        with self._lock:
                            self._reading = LiveReading(True, values, warnings=tuple(visible_warnings))
        except Exception as exc:
            with self._lock:
                self._reading = LiveReading(error=str(exc))
        finally:
            with self._lock:
                reading = self._reading
                self._reading = LiveReading(False, reading.values, reading.error, reading.warnings)


@st.cache_resource
def get_live_sensor():
    reader = LiveSensor()
    atexit.register(reader.close)
    return reader
