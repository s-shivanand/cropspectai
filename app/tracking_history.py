"""Display saved Cab and MNDVI progression for the actively tracked specimen."""

import altair as alt
import pandas as pd
import streamlit as st

from app.tracking import read_tracking_predictions


def load_tracking_history(path, specimen):
    """Keep recorded values intact; exclude unusable data only from the display."""
    rows = read_tracking_predictions(path)
    if not rows:
        return pd.DataFrame(columns=["timestamp", "cab", "mndvi_as7343"])
    data = pd.DataFrame(rows)
    if not {"timestamp", "specimen", "cab"}.issubset(data.columns):
        raise ValueError("The tracking file is missing its timestamp, specimen, or Cab column.")
    if not data["specimen"].eq(specimen).all():
        raise ValueError("The tracking file contains a different specimen.")
    data["timestamp"] = pd.to_datetime(data["timestamp"], format="mixed", errors="coerce")
    for column in ("cab", "mndvi_as7343"):
        data[column] = pd.to_numeric(data.get(column, pd.Series(dtype=float)), errors="coerce")
        data[column] = data[column].replace([float("inf"), float("-inf")], float("nan"))
    return data.dropna(subset=["timestamp"]).sort_values("timestamp", kind="stable")


def render_tracking_history(path, specimen):
    with st.container(key="tracking_history", gap="small"):
        st.subheader("Specimen progression", anchor=False, text_alignment="center")
        st.caption(f"Specimen: {specimen} · Saved tracking history")
        try:
            data = load_tracking_history(path, specimen)
        except (OSError, ValueError) as exc:
            st.warning(f"Could not load tracking history: {exc}")
            return
        if data.empty:
            st.caption("No dated tracking measurements are available yet.")
            return
        if len(data) == 1:
            st.caption("First tracked measurement saved. Further measurements will build the progression.")
        for column, title, axis_title, color in (
            ("cab", "Cab progression", "Predicted Cab (µg/cm²)", "#86efac"),
            ("mndvi_as7343", "MNDVI progression", "MNDVI", "#7dd3fc"),
        ):
            st.markdown(f"**{title}**")
            values = data.dropna(subset=[column])
            if values.empty:
                st.caption(f"No valid {axis_title} values are available in the tracking history.")
            chart = alt.Chart(values).mark_line(
                color=color, strokeWidth=1.5,
                point=alt.OverlayMarkDef(size=65, filled=True, color=color),
            ).encode(
                x=alt.X("timestamp:T", title="Measurement time",
                        axis=alt.Axis(labelAngle=-20, labelOverlap=True, tickCount=4)),
                y=alt.Y(f"{column}:Q", title=axis_title, scale=alt.Scale(zero=False)),
                order=alt.Order("timestamp:T"),
                tooltip=[alt.Tooltip("timestamp:T", title="Measured at", format="%d %b %Y, %H:%M:%S"),
                         alt.Tooltip(f"{column}:Q", title=axis_title, format=".4f")],
            ).properties(
                height=220, background="transparent",
                padding={"top": 12, "right": 12, "bottom": 8, "left": 8},
            ).configure_axis(
                labelColor="#d4e4d9", titleColor="#e5efe8",
                labelFont="Avenir Next", titleFont="Avenir Next",
                labelFontSize=13, titleFontSize=14, titlePadding=12,
                gridColor="#688473", gridOpacity=0.2, domain=False,
            ).configure_view(strokeOpacity=0)
            st.altair_chart(chart, width="stretch", theme=None, key=f"history_{column}")
