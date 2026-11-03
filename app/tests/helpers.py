"""Build app tests with the device workflow selected."""

from streamlit.testing.v1 import AppTest


def device_app(path):
    app = AppTest.from_file(path)
    app.session_state.device_available = True
    return app
