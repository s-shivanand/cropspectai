from cropspectai import config
from cropspectai.data_analysis.preprocessing import split_dataset_into_X_y
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def downsample_hyperspectra_to_channel(features: np.ndarray, wavelengths: np.ndarray, srf: np.ndarray) -> np.ndarray:

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

def downsample_to_as7343(features: pd.DataFrame, wavelengths, srf_df) -> pd.DataFrame:
    spectra = features.to_numpy()
    reflectance = {}

    for channel in config.AS7343_CHANNELS:

        reflectance[channel] = downsample_hyperspectra_to_channel(
            spectra,
            wavelengths,
            srf_df[channel].to_numpy(),
        )

    return pd.DataFrame(reflectance, index=features.index)

def generate_gaussian_srfs(wavelengths):

    srfs = {}

    for channel, params in config.AS7343_CHANNELS.items():
        srfs[channel] = gaussian_srf(
            wavelengths,
            peak=params["peak"],
            fwhm=params["fwhm"],
        )

    #convert to dataframe
    srf_df = pd.DataFrame(srfs,index=wavelengths)
    srf_df.index.name = "wavelength"  

    return srf_df  


def gaussian_srf(wavelengths: np.ndarray, peak: float, fwhm: float) -> np.ndarray:
    sigma = fwhm / config.FWHM_std_factor 
    gaussian_density_values = np.exp(-0.5 * ((wavelengths - peak) / sigma) ** 2)

    return gaussian_density_values


def plot_srfs(srf_df: pd.DataFrame) -> None:
    plt.figure(figsize=(12, 6))

    for channel in srf_df.columns:
        plt.plot(
            srf_df.index,
            srf_df[channel],
            label=channel,
            color=config.channel_colors[channel],
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
    train_path = config.RAW_PARQUET_DATA_SPLITS_PATH / "train.parquet"

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
    # output_path = config.DOWNSAMPLED_PARQUET_DATA_SPLITS_PATH / "train.parquet"
    # downsampled_train.to_parquet(output_path, engine="pyarrow", index=False)
