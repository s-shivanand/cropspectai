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
    """Immutable snapshot of the live reader's display state.

    Attributes:
        running: Whether the background reader is active.
        values: Latest reflectance values in channel order, or None before a
            valid reading is available.
        error: Reader failure message, or None.
        warnings: Displayable calibration or saturation warnings.
    """

    running: bool = False
    values: tuple[float, ...] | None = None
    error: str | None = None
    warnings: tuple[str, ...] = ()


class LiveSensor:
    """Manage one serial reader shared by Streamlit sessions.

    A lock protects ownership and reading state. A background thread acquires
    normalized spectra without saving them, while heartbeat and data timeouts
    release stalled connections. No serial connection is opened on construction.

    Attributes:
        HEARTBEAT_TIMEOUT: Seconds allowed without an owning browser heartbeat.
        DATA_TIMEOUT: Seconds allowed without a valid live data frame.
    """

    HEARTBEAT_TIMEOUT: float = 20
    DATA_TIMEOUT: float = 15

    def __init__(self) -> None:
        """Initialize an idle reader without opening the serial port."""
        self._lock = Lock()
        self._thread: Thread | None = None
        self._stop = Event()
        self._owner: str | None = None
        self._heartbeat: float = 0
        self._reading = LiveReading()

    def start(self, owner: str) -> None:
        """Start live acquisition for the owning browser session.

        Args:
            owner: Session identifier requesting the live reader.

        Raises:
            RuntimeError: Another session already owns a running reader.

        Note:
            Repeated starts by the current owner are ignored. Connection failures
            occur in the worker thread and are exposed through snapshot().
        """
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

    def snapshot(self, owner: str) -> LiveReading:
        """Return the latest state and refresh the owner's heartbeat.

        Args:
            owner: Session identifier requesting the snapshot.

        Returns:
            Current immutable reading for the owner, or an empty idle reading for
            another session. This method does not initiate sensor acquisition.
        """
        with self._lock:
            if owner != self._owner:
                return LiveReading()
            self._heartbeat = time.monotonic()
            return self._reading

    def stop(self, owner: str | None) -> None:
        """Request reader shutdown and wait briefly for its thread to finish.

        Args:
            owner: Session identifier allowed to stop the reader, or None before
                any owner has been assigned. Other sessions are ignored.

        Raises:
            RuntimeError: The reader has not stopped within the 12-second join timeout.

        Note:
            Retains the last spectrum. The worker's connection context stops
            streaming and closes the serial port.
        """
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

    def close(self) -> None:
        """Stop acquisition for the current owner during application cleanup.

        Raises:
            RuntimeError: The background reader does not stop within its join timeout.
        """
        self.stop(self._owner)

    def _read(self) -> None:
        """Read valid live frames until stopped, disconnected, or timed out.

        Note:
            Holds the shared measurement lock while connected. Invalid frames are
            skipped. Errors and visible warnings are stored in reading snapshots;
            the final snapshot is marked stopped while retaining its last values.
        """
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
                            raise TimeoutError(
                                "No live readings received. Check the sensor and try again."
                            )
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
                        visible_warnings = [
                            message
                            for message in warnings
                            if not message.startswith("ESP32 readiness")
                        ]
                        if max(raw) > 60000:
                            visible_warnings.append(
                                "A channel is near saturation. Lower the gain or LED current."
                            )
                        with self._lock:
                            self._reading = LiveReading(
                                True, values, warnings=tuple(visible_warnings)
                            )
        except Exception as exc:
            with self._lock:
                self._reading = LiveReading(error=str(exc))
        finally:
            with self._lock:
                reading = self._reading
                self._reading = LiveReading(False, reading.values, reading.error, reading.warnings)


@st.cache_resource
def get_live_sensor() -> LiveSensor:
    """Return the cached live reader shared by app sessions.

    Returns:
        LiveSensor instance with a process-exit cleanup handler registered on
        creation. Reading begins only when start() is called.
    """
    reader = LiveSensor()
    atexit.register(reader.close)
    return reader
