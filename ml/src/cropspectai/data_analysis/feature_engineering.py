"""Feature engineering for AS7343 reflectance data."""

from __future__ import annotations

from typing import Literal

import numpy as np
import pandas as pd

from cropspectai import ml_config
from cropspectai.data_analysis import preprocessing

AS7343_CHANNELS = list(ml_config.AS7343_CHANNELS.keys())

EPS = 1e-12


def _safe_divide(
    numerator: pd.Series,
    denominator: pd.Series,
    eps: float = EPS,
) -> pd.Series:
    """Divide aligned Series while masking near-zero denominators.

    Args:
        numerator: Numeric numerator values indexed by sample.
        denominator: Numeric denominator values aligned with the numerator.
        eps: Denominators with absolute values at or below this tolerance are
            replaced by NaN before division.

    Returns:
        Floating-point Series aligned by index. Missing or near-zero
        denominator values produce NaN results.
    """
    denominator = denominator.astype(float)
    safe_denominator = denominator.where(denominator.abs() > eps, np.nan)
    return numerator.astype(float) / safe_denominator


def _validate_as7343_channels(X: pd.DataFrame) -> None:
    """Check that every configured AS7343 channel is present.

    Args:
        X: Reflectance DataFrame to inspect. Additional columns are allowed.

    Raises:
        ValueError: One or more required channel columns are missing.
    """
    missing = [channel for channel in AS7343_CHANNELS if channel not in X.columns]

    if missing:
        raise ValueError(f"Missing AS7343 channel columns: {missing}. Expected: {AS7343_CHANNELS}")


def add_established_chlorophyll_indices(X: pd.DataFrame) -> pd.DataFrame:
    """Append established chlorophyll-sensitive indices to reflectance data.

    Args:
        X: Numeric reflectance DataFrame containing all twelve AS7343 channels.
            Other columns are retained in their original order.

    Returns:
        Copy of the input with chlorophyll index columns added or overwritten.
        The input index is retained. Near-zero divisions and infinite values
        become NaN; the input DataFrame is not modified.

    Raises:
        ValueError: A required AS7343 channel is missing.

    Note:
        Names ending in ``_as7343`` indicate approximations using available
        sensor bands: FZ for blue (~450 nm), F5 for green (~550 nm), F7 for
        red (~690 nm), F8 for red edge (~745 nm), and NIR for near infrared
        (~855 nm). These bands do not exactly match every published formula.
    """
    _validate_as7343_channels(X)

    out = X.copy()

    blue = out["FZ"]
    green = out["F5"]
    red = out["F7"]
    red_edge = out["F8"]
    nir = out["NIR"]

    # Chlorophyll Index - Red Edge
    out["ci_red_edge_as7343"] = _safe_divide(nir, red_edge) - 1.0

    # Chlorophyll Index - Green
    out["ci_green_as7343"] = _safe_divide(nir, green) - 1.0

    # Gitelson-Merzlyak red-edge normalized difference.
    # Canonical form uses approximately R750 and R705.
    out["gm_nd_as7343"] = _safe_divide(
        red_edge - red,
        red_edge + red,
    )

    # Red-edge ratio family (approximately R750 / R700).
    out["red_edge_ratio_as7343"] = _safe_divide(red_edge, red)

    # Datt ratio family (approximately R850 / R710).
    out["datt2_as7343"] = _safe_divide(nir, red)

    # Datt difference-ratio family.
    # This is a coarse approximation because AS7343 has no ~710 nm channel.
    out["datt1_as7343"] = _safe_divide(
        nir - red_edge,
        nir - red,
    )

    # Modified Chlorophyll Absorption Ratio Index (MCARI).
    out["mcari_as7343"] = ((red_edge - red) - 0.2 * (red_edge - green)) * _safe_divide(
        red_edge, red
    )

    # Transformed Chlorophyll Absorption Ratio Index (TCARI).
    out["tcari_as7343"] = 3.0 * (
        (red_edge - red) - 0.2 * (red_edge - green) * _safe_divide(red_edge, red)
    )

    # Optimized Soil Adjusted Vegetation Index, used as the denominator in
    # chlorophyll-sensitive MCARI/OSAVI and TCARI/OSAVI combinations.
    out["osavi_as7343"] = 1.16 * _safe_divide(
        nir - red,
        nir + red + 0.16,
    )

    out["mcari_osavi_as7343"] = _safe_divide(
        out["mcari_as7343"],
        out["osavi_as7343"],
    )

    out["tcari_osavi_as7343"] = _safe_divide(
        out["tcari_as7343"],
        out["osavi_as7343"],
    )

    # Modified NDVI chlorophyll-related family.
    out["mndvi_as7343"] = _safe_divide(
        nir - red,
        nir + red - 2.0 * blue,
    )

    out.replace([np.inf, -np.inf], np.nan, inplace=True)

    return out


def add_additional_spectral_features(X: pd.DataFrame) -> pd.DataFrame:
    """Append pigment, greenness, ratio, and sensor-specific features.

    Args:
        X: Numeric reflectance DataFrame containing all twelve AS7343 channels.
            Other columns, including established indices, are retained.

    Returns:
        Copy of the input with additional feature columns added or overwritten.
        The input index is retained. Near-zero divisions and infinite values
        become NaN; the input DataFrame is not modified.

    Raises:
        ValueError: A required AS7343 channel is missing.

    Note:
        Excludes the established chlorophyll-index group so this feature set
        can be evaluated separately. Some features are sensor-specific
        contrasts or ratios rather than canonical published indices.
    """
    _validate_as7343_channels(X)

    out = X.copy()

    blue = out["FZ"]
    green = out["F5"]
    green_broad = out["FY"]
    orange = out["FXL"]
    red = out["F7"]
    red_edge = out["F8"]
    nir = out["NIR"]

    # General vegetation / greenness indices.
    out["ndvi_as7343"] = _safe_divide(nir - red, nir + red)
    out["gndvi_as7343"] = _safe_divide(nir - green, nir + green)
    out["sr_as7343"] = _safe_divide(nir, red)
    out["green_sr_as7343"] = _safe_divide(nir, green)

    out["evi2_as7343"] = 2.5 * _safe_divide(
        nir - red,
        1.0 + nir + 2.4 * red,
    )

    out["rdvi_as7343"] = _safe_divide(
        nir - red,
        np.sqrt(np.clip(nir + red, a_min=0.0, a_max=None)),
    )

    # Pigment / senescence-related indices.
    out["sipi_as7343"] = _safe_divide(
        nir - blue,
        nir - red,
    )

    out["psri_as7343"] = _safe_divide(
        red - out["F4"],
        red_edge,
    )

    # AS7343-native spectral contrasts and ratios.
    # These are engineered features, not canonical published index formulas.
    out["red_green_nd"] = _safe_divide(red - green, red + green)

    out["red_edge_green_nd"] = _safe_divide(
        red_edge - green,
        red_edge + green,
    )

    out["nir_green_nd"] = _safe_divide(
        nir - green,
        nir + green,
    )

    out["red_edge_slope_proxy"] = (red_edge - red) / (745.0 - 690.0)

    out["green_red_ratio"] = _safe_divide(green, red)
    out["rededge_green_ratio"] = _safe_divide(red_edge, green)
    out["nir_fy_ratio"] = _safe_divide(nir, green_broad)
    out["nir_fxl_ratio"] = _safe_divide(nir, orange)

    out.replace([np.inf, -np.inf], np.nan, inplace=True)

    return out


def add_all_features(X: pd.DataFrame) -> pd.DataFrame:
    """Append established chlorophyll indices, then additional features.

    Args:
        X: Numeric reflectance DataFrame containing all twelve AS7343 channels.

    Returns:
        DataFrame retaining the original columns and index, followed by the
        established and additional feature groups. Existing feature names
        are overwritten. The input DataFrame is not modified.

    Raises:
        ValueError: A required AS7343 channel is missing.
    """
    X_features = add_established_chlorophyll_indices(X)
    X_features = add_additional_spectral_features(X_features)

    return X_features


def engineer_dataset_features(
    df: pd.DataFrame,
    feature_set: Literal["established", "additional", "all"] = "established",
    target: str | None = None,
) -> pd.DataFrame:
    """Engineer dataset features while keeping selected targets last.

    Args:
        df: Dataset containing all twelve AS7343 channels and target labels.
        feature_set: Feature group to add: "established", "additional", or "all".
        target: Single target to retain, or None to retain all configured targets.

    Returns:
        DataFrame ordered as original feature columns, engineered features,
        then selected targets. The index is reset to a consecutive RangeIndex.
        Target and configured non-feature columns are excluded from feature
        calculations. The input DataFrame is not modified.

    Raises:
        ValueError: The feature set is unsupported, target selection is invalid,
            or a required target or AS7343 channel is missing.
    """
    X, y = preprocessing.split_dataset_into_X_y(df, target)

    if feature_set == "established":
        X_engineered = add_established_chlorophyll_indices(X)
    elif feature_set == "additional":
        X_engineered = add_additional_spectral_features(X)
    elif feature_set == "all":
        X_engineered = add_all_features(X)
    else:
        raise ValueError("feature_set must be one of: 'established', 'additional', or 'all'.")

    # X is deliberately concatenated before y so all engineered features sit
    # next to the spectral channels rather than after the target labels.
    return pd.concat(
        [
            X_engineered.reset_index(drop=True),
            y.reset_index(drop=True),
        ],
        axis=1,
    )


if __name__ == "__main__":
    target = "cab"
    datasplit = "train"

    path = ml_config.PROCESSED_PARQUET_DATA_SPLITS_PATH / target / f"{datasplit}.parquet"

    df = pd.read_parquet(path)

    # Experiment 1:
    # 12 AS7343 channels + established chlorophyll indices + target columns.
    df_established = engineer_dataset_features(df, feature_set="established", target=target)

    established_path = (
        ml_config.PROCESSED_PARQUET_DATA_SPLITS_PATH
        / target
        / f"{datasplit}_established_features.parquet"
    )

    df_established.to_parquet(established_path, index=False)

    # Experiment 2:
    # 12 AS7343 channels + established chlorophyll indices
    # + all additional spectral features + target columns.
    df_all = engineer_dataset_features(df, feature_set="all", target=target)

    all_features_path = (
        ml_config.PROCESSED_PARQUET_DATA_SPLITS_PATH / target / f"{datasplit}_all_features.parquet"
    )

    df_all.to_parquet(all_features_path, index=False)

    print(f"Original shape: {df.shape}")
    print(f"Established feature shape: {df_established.shape}")
    print(f"All-feature shape: {df_all.shape}")
    print(f"Saved: {established_path}")
    print(f"Saved: {all_features_path}")
