"""Local sensor interface: uv run streamlit run app/streamlit_app.py."""

import sys
from html import escape
from pathlib import Path

import streamlit as st

# Allow the app to import the existing script when launched from Streamlit.
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.measure_predict import measure_and_predict  # noqa: E402

st.set_page_config(page_title="CropSpectAI", page_icon="🍃", layout="centered")

st.markdown(
    """
    <style>
        [data-testid="stAppViewContainer"] {
            background: radial-gradient(ellipse at top, #172d23 0%, #0c1712 65%);
            color: #edf5ef;
            color-scheme: dark;
        }
        [data-testid="stHeader"] {
            background: transparent;
        }
        [data-testid="stMain"],
        [data-testid="stMain"] h1,
        [data-testid="stMain"] h2,
        [data-testid="stMain"] h3,
        [data-testid="stMain"] p,
        [data-testid="stMain"] input,
        [data-testid="stMain"] button,
        [data-testid="stMain"] [data-testid="stMetricValue"] {
            font-family: "Avenir Next", "Segoe UI", sans-serif;
        }
        [data-testid="stMain"] h1 {
            color: #edf5ef;
            font-size: clamp(2.2rem, 6vw, 3.2rem);
            font-weight: 700;
            letter-spacing: -0.04em;
        }
        [data-testid="stMain"] h2,
        [data-testid="stMain"] h3 {
            color: #c8dfcf;
            font-size: 1.35rem;
            font-weight: 600;
            letter-spacing: -0.02em;
        }
        [data-testid="stMain"] [data-testid="stMarkdownContainer"] p {
            color: #d8e6dc;
            font-size: 1rem;
            line-height: 1.65;
        }
        [data-testid="stMain"] [data-testid="stCaptionContainer"] p {
            color: #a5baad;
            font-size: 0.875rem;
        }
        [data-testid="stMain"] [data-testid="stMarkdownContainer"],
        [data-testid="stMain"] h1,
        [data-testid="stMain"] h2,
        [data-testid="stMain"] h3,
        [data-testid="stMain"] p,
        [data-testid="stMain"] [data-testid="stText"],
        [data-testid="stMain"] [data-testid="stMetricValue"] {
            width: 100%;
            text-align: center !important;
        }
        [data-testid="stMain"] [data-testid="stHeadingWithActionElements"],
        [data-testid="stMain"] [data-testid="stWidgetLabel"],
        [data-testid="stMain"] [data-testid="stMetricLabel"],
        [data-testid="stMain"] [data-testid="stSpinner"] {
            justify-content: center;
        }
        [data-testid="stMain"] [data-testid="stForm"] {
            max-width: 400px;
            margin-inline: auto;
            padding: 1.5rem;
            background: #13231c;
            border: 1px solid #304b3c;
            border-radius: 1.25rem;
            box-shadow: 0 12px 32px #00000026;
        }
        [data-testid="stMain"] [data-testid="stTextInputRootElement"] {
            background: #0e1c15;
            border-color: #456250;
            border-radius: 0.75rem;
        }
        [data-testid="stMain"] [data-testid="stTextInputRootElement"]:focus-within {
            border-color: #8bd3a5;
            box-shadow: 0 0 0 2px #8bd3a526;
        }
        [data-testid="stMain"] [data-testid="stTextInput"] input {
            text-align: center;
            color: #edf5ef;
            background: transparent;
        }
        [data-testid="stMain"] [data-testid="stTextInput"] input::placeholder {
            color: #91a99a;
            opacity: 1;
        }
        [data-testid="stMain"] [data-testid="stFormSubmitButton"] button {
            color: #ffffff;
            background: linear-gradient(135deg, #328b5d, #216540);
            border: 1px solid #64af8273;
            border-radius: 1rem;
            padding: 0.8rem 1.6rem;
            box-shadow: 0 6px 18px #00000033, inset 0 1px 0 #ffffff26;
            transition: transform 150ms ease, box-shadow 150ms ease;
        }
        [data-testid="stMain"] [data-testid="stFormSubmitButton"] button p {
            color: inherit;
            font-family: "Trebuchet MS", "Avenir Next", sans-serif;
            font-size: 1.05rem;
            font-weight: 600;
            letter-spacing: 0.015em;
        }
        [data-testid="stMain"] [data-testid="stFormSubmitButton"] button:hover {
            background: linear-gradient(135deg, #3a9b69, #287b4e);
            border-color: #8bd3a5;
            box-shadow: 0 8px 24px #00000040, inset 0 1px 0 #ffffff33;
            transform: translateY(-1px);
        }
        [data-testid="stMain"] [data-testid="stFormSubmitButton"] button:active {
            transform: translateY(0);
            box-shadow: inset 0 2px 5px #00000033;
        }
        [data-testid="stMain"] [data-testid="stFormSubmitButton"] button:focus-visible {
            outline: 2px solid #b1edc5;
            outline-offset: 3px;
        }
        [data-testid="stMain"] [data-testid="stMetricValue"],
        [data-testid="stMain"] [data-testid="stMetricValue"] * {
            color: #b4efca !important;
            font-family: "SF Pro Display", "Avenir Next", "Segoe UI", sans-serif;
            font-size: clamp(2.4rem, 5vw, 3.5rem) !important;
            font-weight: 650;
            letter-spacing: -0.03em;
            font-variant-numeric: tabular-nums;
            line-height: 1.2;
        }
        [data-testid="stMain"] hr {
            border-color: #304b3c;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("🍃 CropSpectAI")
st.write("Measure a leaf’s spectrum and estimate its chlorophyll content.")
st.caption("Connect the ESP32 measuring device and position the leaf before measuring.")

with st.form("measurement", enter_to_submit=False):
    specimen = st.text_input("Specimen label", placeholder="e.g. leaf_01")
    with st.container(horizontal=True, horizontal_alignment="center"):
        submitted = st.form_submit_button("Measure and predict", type="primary")

if submitted:
    if not specimen.strip():
        st.warning("Please enter a specimen label before measuring.")
    else:
        # Clear stale results so a failed attempt cannot look like a new prediction.
        st.session_state.pop("measurement_result", None)
        st.session_state.pop("measurement_error", None)
        try:
            with st.container(horizontal=True, horizontal_alignment="center"):
                with st.spinner(
                    "Connecting to the sensor and measuring. Keep the leaf in place…",
                    width="content",
                ):
                    st.session_state.measurement_result = measure_and_predict(specimen)
        except Exception as exc:
            st.session_state.measurement_error = str(exc)

if st.session_state.get("measurement_error"):
    st.error(st.session_state.measurement_error)

result = st.session_state.get("measurement_result")
if result is not None:
    st.divider()
    st.subheader("Chlorophyll (Cab) measurement")
    # st.text(f"Specimen: {result.measurement['specimen']}")
    # st.caption(f"Measured at {result.measurement['timestamp']}")
    if result.cab is not None:
        st.metric(
            label="Predicted Cab",
            value=f"{result.cab:.4f} µg/cm²",
            label_visibility="collapsed",
        )
    else:
        st.error(f"Prediction failed: {result.prediction_error}. "
                 "The spectral measurement is saved; no prediction was saved.")
    for warning in result.warnings:
        if warning == (
            "ESP32 readiness was not confirmed. If acquisition fails, "
            "press RST on the device and try again."
        ):
            continue
        if warning.startswith("Noisy channels: "):
            channels = escape(warning.removeprefix("Noisy channels: "))
            st.markdown(
                f'<span style="color: #eab308;">Warning: noisy channels at {channels}</span>',
                unsafe_allow_html=True,
            )
        else:
            st.warning(warning)
