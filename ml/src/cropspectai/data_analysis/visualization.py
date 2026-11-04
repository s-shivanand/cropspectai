"""Plot spectral samples, reflectance summaries, and target distributions."""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from cropspectai import ml_config
from cropspectai.data_analysis.preprocessing import split_dataset_into_X_y
from cropspectai.ml_config import RAW_PARQUET_DATA_SPLITS_PATH


def plot_spectra(X: pd.DataFrame, n_samples: int = 500) -> None:
    """Plot a reproducible random sample of reflectance spectra.

    Args:
        X: Spectral DataFrame with numeric wavelength labels or AS7343 channel
            names as columns, and samples as rows.
        n_samples: Maximum number of spectra to plot. Sampling uses seed 42.

    Note:
        Creates a Matplotlib figure with bands sorted by wavelength. Call
        ``plt.show()`` to display it outside an interactive notebook. No figure
        is returned or saved.
    """
    wavelengths, X = obtain_wavelengths_from_data(X)

    sample = X.sample(
        n=min(n_samples, len(X)),
        random_state=42,
    )

    plt.figure(figsize=(12, 6))

    for _, spectrum in sample.iterrows():
        plt.plot(
            wavelengths,
            spectrum.values,
            alpha=1,
        )

    plt.xlabel("Wavelength (nm)")
    plt.ylabel("Reflectance")
    plt.title(f"Random sample of {len(sample)} spectra")
    # plt.show()


def obtain_wavelengths_from_data(X: pd.DataFrame) -> tuple[np.ndarray, pd.DataFrame]:
    """Resolve column wavelengths and sort spectral features accordingly.

    Args:
        X: Spectral DataFrame whose columns are numeric wavelength labels in
            nanometers or configured AS7343 channel names.

    Returns:
        Pair containing a one-dimensional sorted wavelength array and a copy
        of ``X`` with columns reordered to match. The row index is retained.

    Raises:
        KeyError: A nonnumeric column name is not a configured AS7343 channel.
    """
    try:
        # Hyperspectral data: column names are wavelengths
        wavelengths = X.columns.astype(float).to_numpy()

    except ValueError:
        # Downsampled AS7343 data: column names are channel names
        wavelengths = np.array(
            [ml_config.AS7343_CHANNELS[channel]["peak"] for channel in X.columns]
        )

    # Sort by wavelength and corresponding data
    order = np.argsort(wavelengths)
    wavelengths = wavelengths[order]
    X = X.iloc[:, order].copy()

    return wavelengths, X


def plot_spectral_distribution(X: pd.DataFrame) -> None:
    """Plot wavelength-wise reflectance averages and percentile bands.

    Args:
        X: Spectral DataFrame with samples as rows and numeric wavelength labels
            or configured AS7343 channel names as columns.

    Note:
        Creates a Matplotlib figure with mean, median, 5-95%, and 25-75%
        summaries. Call ``plt.show()`` to display it outside a notebook. No
        figure is returned or saved.
    """
    wavelengths, X = obtain_wavelengths_from_data(X)

    median = X.median()
    mean = X.mean()
    q05 = X.quantile(0.05)
    q25 = X.quantile(0.25)
    q75 = X.quantile(0.75)
    q95 = X.quantile(0.95)

    plt.figure(figsize=(12, 6))

    plt.fill_between(
        wavelengths,
        q05,
        q95,
        alpha=0.25,
        label="5–95%",
    )

    plt.fill_between(
        wavelengths,
        q25,
        q75,
        alpha=0.45,
        label="25–75%",
    )

    plt.plot(
        wavelengths,
        median,
        # linewidth=2,
        color="green",
        linestyle="--",
        label="Median",
    )

    plt.plot(
        wavelengths,
        mean,
        # linewidth=2,
        color="red",
        linestyle="-",
        label="Mean",
    )

    plt.xlabel("Wavelength (nm)")
    plt.ylabel("Reflectance")
    plt.title("Spectral distribution")
    plt.legend()
    # plt.show()


def plot_target_distributions(y: pd.DataFrame) -> None:
    """Plot histograms with mean and median markers for target columns.

    Args:
        y: DataFrame whose numeric columns are target variables and whose rows
            are samples.

    Note:
        Creates a Matplotlib figure using 30 bins per histogram. Call
        ``plt.show()`` to display it outside a notebook. No axes or figure are
        returned, and the input is unchanged.
    """
    median = y.median()
    mean = y.mean()

    axes = y.hist(bins=30, figsize=(14, 8), color="steelblue", edgecolor="black", alpha=0.7)

    for ax, col in zip(axes.flatten(), y.columns):
        mean = y[col].mean()
        median = y[col].median()

        ax.axvline(
            mean,
            color="red",
            linestyle="-",
            label=f"Mean: {mean:.3g}",
        )

        ax.axvline(
            median,
            color="green",
            linestyle="--",
            label=f"Median: {median:.3g}",
        )

        ax.legend(fontsize=8)

    plt.tight_layout()
    # plt.show()


def plot_spectra_by_grouped_target(X: pd.DataFrame, y: pd.DataFrame) -> None:
    """Plot mean spectra for low, medium, and high target quantiles.

    Args:
        X: Spectral DataFrame with numeric wavelength labels or configured
            AS7343 channel names as columns. Its index must align with ``y``.
        y: Numeric target DataFrame sharing the sample index of ``X``.

    Raises:
        ValueError: A target cannot be divided into three distinct quantile bins.

    Note:
        Uses a fixed 4-by-2 subplot grid, displaying at most eight targets, and
        marks the 1000 nm sensor boundary. Call ``plt.show()`` to display the
        figure outside a notebook. No figure is returned or saved.
    """
    group_order = ["Low", "Medium", "High"]

    line_styles = {
        "Low": "-",
        "Medium": "--",
        "High": ":",
    }

    wavelengths, X = obtain_wavelengths_from_data(X)

    # 4 rows × 2 columns for 8 targets
    fig, axes = plt.subplots(
        nrows=4,
        ncols=2,
        figsize=(16, 10),
        sharex=True,
        sharey=True,
    )

    axes = axes.flatten()

    for ax, target in zip(axes, y.columns):
        target_values = y[target]

        groups = pd.qcut(
            target_values,
            q=3,
            labels=group_order,
        )

        for group in group_order:
            mask = groups == group

            mean_spectrum = X.loc[mask].mean()

            group_min = target_values[mask].min()
            group_max = target_values[mask].max()

            ax.plot(
                wavelengths,
                mean_spectrum,
                linestyle=line_styles[group],
                label=f"{group} ({group_min:.3g}–{group_max:.3g})",
            )

        # Shade sensor range
        ax.axvspan(
            wavelengths.min(),
            1000,
            alpha=0.25,
            label="Sensor range",
        )

        # Sensor limit
        ax.axvline(
            x=1000,
            color="red",
            linestyle="--",
            linewidth=1.5,
            label="Sensor limit (1000 nm)",
        )

        ax.set_title(target)
        ax.set_xlabel("Wavelength (nm)")
        ax.set_ylabel("Mean reflectance")
        ax.legend()

    fig.suptitle(
        "Mean Spectra by Grouped Target",
        fontsize=12,
    )

    fig.tight_layout(h_pad=3.0)
    fig.subplots_adjust(hspace=0.4)


if __name__ == "__main__":
    train_path = RAW_PARQUET_DATA_SPLITS_PATH / "train.parquet"
    df_train = pd.read_parquet(train_path)
    X_train, y_train = split_dataset_into_X_y(df_train)

    # plot input features
    plot_spectra(X_train)
    plot_spectral_distribution(X_train)

    # plot targets
    plot_target_distributions(y_train)
    plot_spectra_by_grouped_target(X_train, y_train)

    plt.show()
