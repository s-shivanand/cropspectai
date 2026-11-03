"""Feature engineering for AS7343 reflectance data."""

from __future__ import annotations

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
    """Element-wise division while avoiding division by values near zero."""
    denominator = denominator.astype(float)
    safe_denominator = denominator.where(denominator.abs() > eps, np.nan)
    return numerator.astype(float) / safe_denominator


def _validate_as7343_channels(X: pd.DataFrame) -> None:
    """Check that all required AS7343 reflectance channels are present."""
    missing = [channel for channel in AS7343_CHANNELS if channel not in X.columns]

    if missing:
        raise ValueError(
            f"Missing AS7343 channel columns: {missing}. "
            f"Expected: {AS7343_CHANNELS}"
        )


def add_established_chlorophyll_indices(X: pd.DataFrame) -> pd.DataFrame:
    """Add established chlorophyll-sensitive vegetation indices.

    The original AS7343 channel columns are retained and the new index columns
    are appended immediately after them.

    Notes
    -----
    Several published indices require narrow bands around wavelengths that the
    AS7343 does not measure exactly. Their names therefore use an ``_as7343``
    suffix to make clear that the nearest useful AS7343 channels are used.

    Approximate mapping:
        blue     -> FZ  (~450 nm)
        green    -> F5  (~550 nm)
        red      -> F7  (~690 nm)
        red-edge -> F8  (~745 nm)
        NIR      -> NIR (~855 nm)
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
    out["mcari_as7343"] = (
        (red_edge - red) - 0.2 * (red_edge - green)
    ) * _safe_divide(red_edge, red)

    # Transformed Chlorophyll Absorption Ratio Index (TCARI).
    out["tcari_as7343"] = 3.0 * (
        (red_edge - red)
        - 0.2 * (red_edge - green) * _safe_divide(red_edge, red)
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
    """Add secondary pigment, greenness, ratio, and AS7343-native features.

    This function intentionally excludes the established chlorophyll-index
    group above. It can therefore be evaluated as a separate feature set.
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

    out["red_edge_slope_proxy"] = (
        (red_edge - red) / (745.0 - 690.0)
    )

    out["green_red_ratio"] = _safe_divide(green, red)
    out["rededge_green_ratio"] = _safe_divide(red_edge, green)
    out["nir_fy_ratio"] = _safe_divide(nir, green_broad)
    out["nir_fxl_ratio"] = _safe_divide(nir, orange)

    out.replace([np.inf, -np.inf], np.nan, inplace=True)

    return out


def add_all_features(X: pd.DataFrame) -> pd.DataFrame:
    """Add established chlorophyll indices followed by additional features."""
    X_features = add_established_chlorophyll_indices(X)
    X_features = add_additional_spectral_features(X_features)

    return X_features


def engineer_dataset_features(
    df: pd.DataFrame,
    feature_set: str = "established", target: str = None
) -> pd.DataFrame:
    """Feature-engineer a full dataset while keeping target columns last.

    The dataframe is first split into X and y using the project's existing
    preprocessing helper. Features are generated only from X, preventing
    target columns from entering any feature calculations.

    Parameters
    ----------
    df:
        Full dataframe containing AS7343 channels and target label columns.
    feature_set:
        "established" -> established chlorophyll indices only.
        "additional"  -> additional spectral features only.
        "all"         -> established indices followed by additional features.

    Returns
    -------
    pd.DataFrame
        Columns ordered as:
        [original X channels/features] -> [new engineered features] -> [targets]
    """
    X, y = preprocessing.split_dataset_into_X_y(df, target)

    if feature_set == "established":
        X_engineered = add_established_chlorophyll_indices(X)
    elif feature_set == "additional":
        X_engineered = add_additional_spectral_features(X)
    elif feature_set == "all":
        X_engineered = add_all_features(X)
    else:
        raise ValueError(
            "feature_set must be one of: "
            "'established', 'additional', or 'all'."
        )

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

    path = (
        ml_config.PROCESSED_PARQUET_DATA_SPLITS_PATH
        / target
        / f"{datasplit}.parquet"
    )

    df = pd.read_parquet(path)

    # Experiment 1:
    # 12 AS7343 channels + established chlorophyll indices + target columns.
    df_established = engineer_dataset_features(
        df,
        feature_set="established", target=target)

    established_path = (
        ml_config.PROCESSED_PARQUET_DATA_SPLITS_PATH
        / target
        / f"{datasplit}_established_features.parquet"
    )

    df_established.to_parquet(established_path, index=False)

    # Experiment 2:
    # 12 AS7343 channels + established chlorophyll indices
    # + all additional spectral features + target columns.
    df_all = engineer_dataset_features(
        df,
        feature_set="all", target=target
    )

    all_features_path = (
        ml_config.PROCESSED_PARQUET_DATA_SPLITS_PATH
        / target
        / f"{datasplit}_all_features.parquet"
    )

    df_all.to_parquet(all_features_path, index=False)

    print(f"Original shape: {df.shape}")
    print(f"Established feature shape: {df_established.shape}")
    print(f"All-feature shape: {df_all.shape}")
    print(f"Saved: {established_path}")
    print(f"Saved: {all_features_path}")
