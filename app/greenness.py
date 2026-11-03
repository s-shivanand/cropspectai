"""Display the existing AS7343 MNDVI feature on a fixed greenness scale."""

import math

import pandas as pd

from cropspectai import ml_config
from cropspectai.data_analysis.feature_engineering import add_established_chlorophyll_indices

SPECTRUM_HEIGHT = 280
GAUGE_WIDTH = 120
GAUGE_TOP = 25  # Change this to move the bar, ticks, pointer, and labels together.


def spectrum_mndvi(values):
    """Use the same channel mapping and safe division as model feature engineering."""
    frame = pd.DataFrame([values], columns=list(ml_config.AS7343_CHANNELS))
    value = float(add_established_chlorophyll_indices(frame)["mndvi_as7343"].iloc[0])
    return value if math.isfinite(value) else None


def greenness_gauge(value):
    """Render a yellow-to-green scale, with a bounded marker and actual MNDVI."""
    available = value is not None and math.isfinite(value)
    actual = f"{value:.3f}" if available else "N/A"
    note = ""
    marker = ""
    bottom = GAUGE_TOP + SPECTRUM_HEIGHT
    svg_height = bottom + 94
    if available:
        position = GAUGE_TOP + (1 - min(1, max(0, value))) * SPECTRUM_HEIGHT
        marker = (
            f'<path d="M 10 {position - 6:.2f} L 21 {position:.2f} '
            f'L 10 {position + 6:.2f} Z" fill="#edf5ef"/>'
        )
        if value > 1:
            note = "Above 0–1 scale"
        elif value < 0:
            note = "Below 0–1 scale"
    else:
        note = "Index unavailable"
    tick_positions = [
        (GAUGE_TOP + (1 - level) * SPECTRUM_HEIGHT, f"{level:.2f}")
        for level in (1, 0.75, 0.5, 0.25, 0)
    ]
    ticks = "".join(
        f'<line x1="62" y1="{y}" x2="68" y2="{y}" stroke="#91a99a"/>'
        f'<text x="75" y="{y + 5}" fill="#e5efe8" font-size="14">{label}</text>'
        for y, label in tick_positions
    )
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{GAUGE_WIDTH}" height="{svg_height}" '
        f'viewBox="0 0 {GAUGE_WIDTH} {svg_height}" '
        f'role="img" aria-label="Relative leaf greenness. MNDVI {actual}. {note}" '
        'style="font-family:Avenir Next,Segoe UI,sans-serif;">'
        '<defs><linearGradient id="mndvi-gradient" x1="0" y1="1" x2="0" y2="0">'
        '<stop offset="0%" stop-color="#f1d64c"/>'
        '<stop offset="10%" stop-color="#e2d34a"/>'
        '<stop offset="25%" stop-color="#155b2b"/>'
        '<stop offset="50%" stop-color="#06351c"/>'
        '<stop offset="75%" stop-color="#042716"/>'
        '<stop offset="100%" stop-color="#021a0e"/>'
        '</linearGradient></defs>'
        f'<rect x="24" y="{GAUGE_TOP}" width="28" height="{SPECTRUM_HEIGHT}" rx="9" '
        'fill="url(#mndvi-gradient)" stroke="#bdd0c3" stroke-opacity="0.35"/>'
        f'{ticks}{marker}'
        f'<text x="60" y="{bottom + 30}" text-anchor="middle" fill="#edf5ef" '
        f'font-size="18" font-weight="600">MNDVI {actual}</text>'
        f'<text x="60" y="{bottom + 51}" text-anchor="middle" fill="#c4f1d4" '
        'font-size="15" font-weight="600">Leaf</text>'
        f'<text x="60" y="{bottom + 69}" text-anchor="middle" fill="#c4f1d4" '
        'font-size="15" font-weight="600">greenness</text>'
        f'<text x="60" y="{bottom + 87}" text-anchor="middle" fill="#eab308" '
        f'font-size="11">{note}</text></svg>'
    )
