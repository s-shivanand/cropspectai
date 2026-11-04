"""Build app tests with the device workflow selected."""

from pathlib import Path

from streamlit.testing.v1 import AppTest


def device_app(path: str | Path) -> AppTest:
    """Create an app test with the device workflow selected but not yet run.

    Args:
        path: Streamlit app script to load.

    Returns:
        AppTest ready for run(), with device_available set to True.
    """
    app = AppTest.from_file(path)
    app.session_state.device_available = True
    return app
