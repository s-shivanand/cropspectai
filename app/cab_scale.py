"""Presentation-only Cab scale; prediction values are never altered."""

import math

TOOLSRTM_RANGE_URL: str = (
    "https://rtm-suite.wur.nl/quarto-hub-toolsrtm/_site/tutorials/python/"
    "t02-parameters-traits.html"
)


def cab_scale(value: float | None) -> str:
    """Render the predicted Cab value on an SVG reference scale.

    Args:
        value: Predicted Cab in micrograms per square centimeter, or None when
            unavailable. Nonfinite values are also treated as unavailable.

    Returns:
        Accessible SVG markup with a 0-100 scale and a highlighted 20-80 band.
        The marker is bounded to the scale, while the displayed value remains
        unchanged and out-of-range predictions are labeled.
    """
    available = value is not None and math.isfinite(value)
    actual = f"{value:.1f}" if available else "unavailable"
    marker = ""
    note = "Cab (µg/cm²)"
    if available:
        position = 24 + min(100, max(0, value)) / 100 * 432
        marker = (
            f'<path d="M {position - 6:.2f} 48 L {position:.2f} 37 '
            f'L {position + 6:.2f} 48 Z" fill="#edf5ef"/>'
        )
        if value < 0 or value > 100:
            note = "Prediction outside the displayed 0–100 scale"
    ticks = "".join(
        f'<line x1="{24 + level * 4.32:.2f}" y1="54" '
        f'x2="{24 + level * 4.32:.2f}" y2="59" stroke="#91a99a"/>'
        f'<text x="{24 + level * 4.32:.2f}" y="75" text-anchor="middle" '
        f'fill="#d4e4d9" font-size="14">{level}</text>'
        for level in (0, 20, 40, 60, 80, 100)
    )
    return (
        '<svg xmlns="http://www.w3.org/2000/svg" width="480" height="103" '
        'viewBox="0 0 480 103" role="img" '
        f'aria-label="Predicted Cab {actual} µg/cm². Scale 0 to 100; '
        '20 to 80 highlighted as a broad typical range for healthy vegetation." '
        'style="font-family:Avenir Next,Segoe UI,sans-serif;">'
        '<rect x="24" y="12" width="432" height="20" rx="10" fill="#293e32"/>'
        '<rect x="110.4" y="12" width="259.2" height="20" fill="#48a36b"/>'
        '<rect x="24" y="12" width="432" height="20" rx="10" '
        'fill="none" stroke="#bdd0c3" stroke-opacity="0.4"/>'
        '<line x1="110.4" y1="9" x2="110.4" y2="35" stroke="#b4efca"/>'
        '<line x1="369.6" y1="9" x2="369.6" y2="35" stroke="#b4efca"/>'
        f"{marker}{ticks}"
        f'<text x="240" y="97" text-anchor="middle" fill="#d4e4d9" '
        f'font-size="14">{note}</text></svg>'
    )
