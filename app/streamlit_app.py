"""Local sensor interface: uv run streamlit run app/streamlit_app.py."""

import sys
from html import escape
from pathlib import Path
from uuid import uuid4

import altair as alt
import pandas as pd
import streamlit as st

# Allow the app to import the existing script when launched from Streamlit.
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.measure_predict import measure_and_predict  # noqa: E402
from scripts import measure_predict as workflow  # noqa: E402
from app.live_sensor import get_live_sensor  # noqa: E402
from app.greenness import GAUGE_WIDTH, SPECTRUM_HEIGHT, greenness_gauge, spectrum_mndvi  # noqa: E402
from app.cab_scale import TOOLSRTM_RANGE_URL, cab_scale  # noqa: E402

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
        [data-testid="stMainBlockContainer"] {
            padding-top: 2rem;
            padding-bottom: 2rem;
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
            font-size: clamp(2.4rem, 5vw, 3.2rem);
            font-weight: 700;
            letter-spacing: -0.04em;
            padding-bottom: 0.25rem;
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
            line-height: 1.45;
        }
        [data-testid="stMain"] [data-testid="stCaptionContainer"] p {
            color: #bdd0c3;
            font-size: 0.875rem;
        }
        [data-testid="stMain"] .st-key-app_intro [data-testid="stMarkdownContainer"] p {
            color: #e5efe8;
            font-size: 1.125rem;
            line-height: 1.5;
        }
        [data-testid="stMain"] .st-key-app_intro [data-testid="stCaptionContainer"] p,
        [data-testid="stMain"] .st-key-reading_controls [data-testid="stCaptionContainer"] p {
            color: #cfdfd3;
            font-size: 1rem;
            line-height: 1.5;
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
            padding: 1.1rem;
            background: #13231c;
            border: 1px solid #304b3c;
            border-radius: 1.25rem;
            box-shadow: 0 12px 32px #00000026;
        }
        [data-testid="stMain"] [data-testid="stForm"] [data-testid="stWidgetLabel"] p {
            color: #e5efe8;
            font-size: 1.125rem;
            font-weight: 500;
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
            font-size: 1.125rem;
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
            font-size: 1.125rem;
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
        [data-testid="stMain"] [data-testid="stFormSubmitButton"] button:focus-visible,
        [data-testid="stMain"] [data-testid="stButton"] button:focus-visible {
            outline: 2px solid #b1edc5;
            outline-offset: 3px;
        }
        [data-testid="stMain"] [data-testid="stButton"] button[kind="secondary"] {
            color: #c8dfcf;
            background: transparent;
            border: 1px solid #648570;
            border-radius: 0.75rem;
            min-width: 6rem;
            padding: 0.45rem 1.15rem;
            box-shadow: none;
        }
        [data-testid="stMain"] [data-testid="stButton"] button[kind="secondary"] p {
            color: inherit;
            font-size: 0.95rem;
            font-weight: 500;
        }
        [data-testid="stMain"] [data-testid="stButton"] button[kind="secondary"]:hover {
            background: #ffffff0a;
            border-color: #9bc5ab;
        }
        [data-testid="stMain"] [data-testid="stMetricValue"],
        [data-testid="stMain"] [data-testid="stMetricValue"] * {
            color: #b4efca !important;
            font-family: "Avenir Next", "Segoe UI", sans-serif;
            font-size: clamp(2.4rem, 5vw, 3.5rem) !important;
            font-weight: 650;
            letter-spacing: -0.03em;
            font-variant-numeric: tabular-nums;
            line-height: 1.2;
        }
        [data-testid="stMain"] hr {
            border-color: #304b3c;
            margin-block: 0.65rem;
        }
        [data-testid="stMain"] .st-key-result_card {
            max-width: 520px;
            margin-inline: auto;
            padding: 1.2rem;
            border: 1px solid #3c5d49;
            border-radius: 1.1rem;
            background: #13231c;
        }
        [data-testid="stMain"] .result-meta {
            color: #cfdfd3;
            font-size: 1rem;
            text-align: center;
            line-height: 1.6;
        }
        [data-testid="stMain"] .st-key-result_card h3 {
            font-size: 1.5rem;
            color: #e5efe8;
        }
        [data-testid="stMain"] .st-key-result_card [data-testid="stCaptionContainer"] p {
            color: #cfdfd3;
            font-size: 1rem;
            line-height: 1.5;
        }
        [data-testid="stMain"] .st-key-result_card [data-testid="stMarkdownContainer"] span {
            font-size: 1.0625rem;
            line-height: 1.5;
        }
        .st-key-spectrum_panel {
            max-height: 0;
            opacity: 0;
            overflow: hidden;
            transition: max-height 450ms ease-in-out, opacity 300ms ease;
        }
        .st-key-spectrum_panel:has(.spectrum-ready) {
            max-height: 640px;
            opacity: 1;
        }
        .st-key-spectrum_panel [data-testid="stImage"] {
            min-height: 399px;
        }
        .st-key-spectrum_status {
            transform: translateY(-2rem);
            min-height: 2.5rem;
            overflow: visible;
            text-align: center;
        }
        @keyframes result-appear {
            from { opacity: 0; transform: translateY(6px); }
            to { opacity: 1; transform: translateY(0); }
        }
        .st-key-result_card {
            animation: result-appear 300ms ease-out;
        }
        @media (max-width: 640px) {
            .st-key-spectrum_panel:has(.spectrum-ready) { max-height: 1200px; }
        }
        @media (prefers-reduced-motion: reduce) {
            .st-key-spectrum_panel { transition: none; }
            .st-key-result_card { animation: none; }
        }
    </style>
    """,
    unsafe_allow_html=True,
)

with st.container(key="app_intro", gap="small"):
    st.title("🍃 CropSpectAI")
    st.write("Measure a leaf’s spectrum and estimate its chlorophyll content.")
    st.caption("Connect the ESP32 measuring device and position the leaf before measuring.")

live_sensor = get_live_sensor()
if "sensor_session" not in st.session_state:
    st.session_state.sensor_session = uuid4().hex
sensor_session = st.session_state.sensor_session


def reset_app():
    """Return this app session to its initial view without changing saved files."""
    try:
        live_sensor.stop(st.session_state.sensor_session)
    except Exception as exc:
        st.session_state.measurement_error = f"Could not reset the app: {exc}"
        return
    for key in (
        "specimen_label", "measurement_result", "measurement_error",
        "averaged_spectrum", "live_requested", "live_action_error", "last_displayed_spectrum",
    ):
        st.session_state.pop(key, None)
    # A fresh owner hides any last frame or error retained by the stopped reader.
    st.session_state.sensor_session = uuid4().hex


@st.fragment(run_every=0.5 if st.session_state.get("live_requested") else None)
def reading_sensor():
    requested = st.session_state.get("live_requested", False)
    reading = live_sensor.snapshot(sensor_session)
    averaged_spectrum = st.session_state.get("averaged_spectrum")
    show_average = averaged_spectrum is not None and not reading.running
    result = st.session_state.get("measurement_result")
    with st.container(key="reading_controls", horizontal_alignment="center", gap="small"):
        if st.button("Stop" if requested else "Start",
                     key="toggle_live", type="secondary"):
            try:
                if requested:
                    live_sensor.stop(sensor_session)
                else:
                    live_sensor.start(sensor_session)
                    st.session_state.pop("averaged_spectrum", None)
                st.session_state.live_requested = not requested
                st.session_state.pop("live_action_error", None)
            except Exception as exc:
                st.session_state.live_action_error = str(exc)
            st.rerun()
        if requested:
            action_hint = "Stop reading to pause the live preview."
        elif show_average:
            action_hint = "Start reading to return to the live preview."
        elif reading.values is not None:
            action_hint = "Start reading to resume the live spectrum preview."
        else:
            action_hint = "Start reading to preview the leaf’s spectrum before measuring."
        st.caption(action_hint)

    if requested and not reading.running:
        st.session_state.live_requested = False
        st.rerun()
    if st.session_state.get("live_action_error"):
        st.error(st.session_state.live_action_error)
    if reading.error:
        st.error(reading.error)
    plot_values = averaged_spectrum if show_average else reading.values
    if plot_values is not None:
        st.session_state.last_displayed_spectrum = tuple(plot_values)
    elif requested and reading.running:
        plot_values = st.session_state.get("last_displayed_spectrum")
    with st.container(key="spectrum_panel", gap="small"):
        st.markdown(
            '<span class="spectrum-ready" hidden></span>' if plot_values is not None
            else '<span hidden></span>',
            unsafe_allow_html=True,
        )
        if plot_values is not None:
            channels = workflow.channel_config
            data = pd.DataFrame({
                "Wavelength (nm)": channels.WAVELENGTHS,
                "Reflectance": plot_values,
                "Channel": channels.CHANNEL_NAMES,
                "Color": channels.BAR_COLORS,
            }).sort_values("Wavelength (nm)")
            data["Wavelength (nm)"] = data["Wavelength (nm)"].astype(str)
            labels = data["Wavelength (nm)"].tolist()
            chart = alt.Chart(data).mark_bar(
                cornerRadiusTopLeft=5, cornerRadiusTopRight=5, opacity=0.9,
            ).encode(
                x=alt.X(
                    "Wavelength (nm):N", sort=labels,
                    scale=alt.Scale(paddingInner=0.28, paddingOuter=0.2),
                    axis=alt.Axis(labelAngle=0, grid=False, titlePadding=16),
                ),
                y=alt.Y(
                    "Reflectance:Q", scale=alt.Scale(domain=[0, 1]),
                    axis=alt.Axis(values=[0, 0.2, 0.4, 0.6, 0.8, 1], format=".1f", titlePadding=14),
                ),
                color=alt.Color("Color:N", scale=None, legend=None),
                tooltip=["Channel:N", "Wavelength (nm):N", alt.Tooltip("Reflectance:Q", format=".4f")],
            ).properties(
                height=SPECTRUM_HEIGHT, background="transparent",
                autosize=alt.AutoSizeParams(type="fit-x", contains="content"),
                padding={"top": 16, "right": 0, "bottom": 16, "left": 0},
            ).configure_axis(
                labelColor="#d4e4d9", titleColor="#e5efe8",
                labelFont="Avenir Next", titleFont="Avenir Next",
                labelFontSize=14, titleFontSize=15, titleFontWeight=600,
                labelPadding=8, domain=False, ticks=False,
                gridColor="#688473", gridOpacity=0.25, gridDash=[3, 5],
            ).configure_view(
                fill="#102018", stroke="#304b3c", strokeOpacity=0.5, cornerRadius=10,
            )
            plot_column, greenness_column = st.columns([6, 1.3], gap="small")
            with plot_column:
                st.altair_chart(chart, width="stretch", theme=None, key="spectrum_plot")
            with greenness_column:
                st.image(
                    greenness_gauge(spectrum_mndvi(plot_values)), width=GAUGE_WIDTH,
                    alt="Relative leaf greenness: MNDVI on a yellow-to-green scale from 0 to 1.",
                )
    if plot_values is not None:
        if show_average:
            count = workflow.as7343_measure.SENSOR_SETTINGS["n_avg"]
            status = f"Average of {count} spectral readings"
            if result is not None:
                status += f" · Specimen: {result.measurement['specimen']}"
            status_color = "#7dd3fc"
        else:
            status = ("Connecting · previous spectrum" if reading.running and reading.values is None
                      else "Reading active" if reading.running else "Reading stopped · last live spectrum")
            status_color = "#86efac" if reading.running else "#eab308"
        status_style = "font-weight: 400; font-size: 1.125rem; line-height: 1.5;"
        with st.container(key="spectrum_status", gap="small"):
            st.markdown(
                f'<div style="color: {status_color}; {status_style} '
                f'text-align: center;">{escape(status)}</div>',
                unsafe_allow_html=True,
            )
    if plot_values is None and reading.running:
        st.caption("Connecting to the sensor and waiting for live readings…")
    for warning in reading.warnings:
        st.warning(warning)


reading_sensor()
st.divider()

with st.form("measurement", enter_to_submit=False):
    specimen = st.text_input("Specimen label", placeholder="e.g. leaf_01", key="specimen_label")
    with st.container(horizontal=True, horizontal_alignment="center"):
        submitted = st.form_submit_button("Measure and predict", type="primary", key="measure_predict")

with st.container(horizontal=True, horizontal_alignment="center"):
    st.button("Reset", key="reset_app", on_click=reset_app,
              help="Stop reading and clear the screen. Saved measurements are kept.")

if submitted:
    if not specimen.strip():
        st.warning("Please enter a specimen label before measuring.")
    else:
        # Clear stale results so a failed attempt cannot look like a new prediction.
        st.session_state.pop("measurement_result", None)
        st.session_state.pop("measurement_error", None)
        st.session_state.pop("averaged_spectrum", None)
        try:
            count = workflow.as7343_measure.SENSOR_SETTINGS["n_avg"]
            with st.container(horizontal=True, horizontal_alignment="center"):
                with st.spinner(
                    f"Taking {count} readings and calculating their average. Keep the leaf in place…",
                    width="content",
                ):
                    live_sensor.stop(sensor_session)
                    st.session_state.live_requested = False
                    st.session_state.measurement_result = measure_and_predict(specimen)
                    st.session_state.averaged_spectrum = tuple(
                        st.session_state.measurement_result.measurement[name]
                        for name in workflow.channel_config.CHANNEL_NAMES
                    )
        except Exception as exc:
            st.session_state.measurement_error = str(exc)
        st.rerun()

if st.session_state.get("measurement_error"):
    st.error(st.session_state.measurement_error)

result = st.session_state.get("measurement_result")
if result is not None:
    with st.container(key="result_card", border=True, gap="small"):
        st.subheader("Predicted Cab", anchor=False, text_alignment="center")
        specimen_label = escape(str(result.measurement["specimen"]))
        measured_at = escape(str(result.measurement["timestamp"]).replace("T", " "))
        st.markdown(
            f'<div class="result-meta">Specimen: <strong>{specimen_label}</strong>'
            f'<br>Measured at {measured_at}</div>',
            unsafe_allow_html=True,
        )
        if result.cab is not None:
            st.metric(
                label="Predicted Cab",
                value=f"{result.cab:.4f} µg/cm²",
                label_visibility="collapsed",
            )
            st.image(
                cab_scale(result.cab), width="stretch",
                alt="Cab scale from 0 to 100 µg/cm², with 20–80 highlighted and the prediction marked.",
            )
            st.caption(
                f"[ToolsRTM]({TOOLSRTM_RANGE_URL}): 20–80 µg/cm² is a broad typical "
                "range for healthy vegetation."
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
