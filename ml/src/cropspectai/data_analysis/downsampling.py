"""Approximate AS7343 reflectance by integrating hyperspectra over Gaussian SRFs."""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from cropspectai import ml_config
from cropspectai.data_analysis.preprocessing import split_dataset_into_X_y


def downsample_hyperspectra_to_channel(
    features: np.ndarray, wavelengths: np.ndarray, srf: np.ndarray
) -> np.ndarray:
    """Integrate spectra against one sensor spectral response function.

    Args:
        features: Numeric array of shape (n_samples, n_wavelengths).
        wavelengths: One-dimensional numeric array of wavelength coordinates
            in nanometers, aligned with the feature columns.
        srf: One-dimensional response array aligned with ``wavelengths``.

    Returns:
        One response-weighted reflectance value per sample, with shape
        (n_samples,). Trapezoidal integrals are normalized by the SRF integral.
        A zero SRF integral is not guarded against and can yield nonfinite values.
    """
    numerator = np.trapezoid(
        features * srf,
        x=wavelengths,
        axis=1,
    )

    denominator = np.trapezoid(
        srf,
        x=wavelengths,
    )

    return numerator / denominator


def downsample_to_as7343(
    features: pd.DataFrame, wavelengths: np.ndarray, srf_df: pd.DataFrame
) -> pd.DataFrame:
    """Downsample hyperspectral features to the configured AS7343 channels.

    Args:
        features: Spectral DataFrame with samples as rows and wavelength bands
            as columns, in the same order as ``wavelengths``.
        wavelengths: One-dimensional numeric array of wavelength coordinates
            in nanometers.
        srf_df: Response DataFrame with one row per wavelength and a column
            for each configured AS7343 channel.

    Returns:
        Reflectance DataFrame retaining the sample index, with columns in the
        configured AS7343 channel order.

    Raises:
        KeyError: A configured channel is missing from ``srf_df``.
    """
    spectra = features.to_numpy()
    reflectance = {}

    for channel in ml_config.AS7343_CHANNELS:
        reflectance[channel] = downsample_hyperspectra_to_channel(
            spectra,
            wavelengths,
            srf_df[channel].to_numpy(),
        )

    return pd.DataFrame(reflectance, index=features.index)


def generate_gaussian_srfs(wavelengths: np.ndarray) -> pd.DataFrame:
    """Generate a Gaussian response for every configured AS7343 channel.

    Args:
        wavelengths: One-dimensional numeric array of wavelengths in nanometers.

    Returns:
        DataFrame indexed by the supplied wavelengths, with one response column
        per configured channel. The index is named "wavelength". Responses are
        sampled Gaussian curves and are not normalized to unit integral.
    """
    srfs = {}

    for channel, params in ml_config.AS7343_CHANNELS.items():
        srfs[channel] = gaussian_srf(
            wavelengths,
            peak=params["peak"],
            fwhm=params["fwhm"],
        )

    # convert to dataframe
    srf_df = pd.DataFrame(srfs, index=wavelengths)
    srf_df.index.name = "wavelength"

    return srf_df


def gaussian_srf(wavelengths: np.ndarray, peak: float, fwhm: float) -> np.ndarray:
    """Evaluate a Gaussian spectral response curve.

    Args:
        wavelengths: Numeric array of wavelengths in nanometers.
        peak: Center wavelength in nanometers.
        fwhm: Positive full width at half maximum in nanometers.

    Returns:
        Response array with the same shape as ``wavelengths``. The curve has a
        theoretical peak of one and is not normalized to unit integral.
    """
    sigma = fwhm / ml_config.FWHM_std_factor
    gaussian_density_values = np.exp(-0.5 * ((wavelengths - peak) / sigma) ** 2)

    return gaussian_density_values


def plot_srfs(srf_df: pd.DataFrame) -> None:
    """Display the configured channels' spectral response curves.

    Args:
        srf_df: Response DataFrame indexed by wavelength in nanometers, with
            AS7343 channel names as columns.

    Raises:
        KeyError: A channel column has no colour in the channel configuration.

    Note:
        Creates a Matplotlib figure and calls ``plt.show()``. No figure is
        returned or saved.
    """
    plt.figure(figsize=(12, 6))

    for channel in srf_df.columns:
        plt.plot(
            srf_df.index,
            srf_df[channel],
            label=channel,
            color=ml_config.channel_colors[channel],
        )
    plt.xlim(left=400, right=1000)
    plt.xlabel("Wavelength (nm)")
    plt.ylabel("Relative Spectral Response")
    plt.title("AS7343 Gaussian Spectral Response Functions")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    train_path = ml_config.RAW_PARQUET_DATA_SPLITS_PATH / "train.parquet"

    df_train = pd.read_parquet(train_path)

    X_train, y_train = split_dataset_into_X_y(df_train)

    wavelengths = X_train.columns.astype(int).to_numpy()
    srf_df = generate_gaussian_srfs(wavelengths)
    print(f"SRF dataframe of train features: \n {srf_df}")

    plot_srfs(srf_df)

    reflectance_df = downsample_to_as7343(X_train, wavelengths, srf_df)
    print(f"\nDownsampled reflectance of train features: \n {reflectance_df}")

    # #save the downsampled data to a parquet file
    # downsampled_train = pd.concat([reflectance_df, y_train], axis=1)
    # output_path = ml_config.DOWNSAMPLED_PARQUET_DATA_SPLITS_PATH / "train.parquet"
    # downsampled_train.to_parquet(output_path, engine="pyarrow", index=False)
