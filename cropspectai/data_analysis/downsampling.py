from cropspectai.config import RAW_PARQUET_DATA_SPLITS_PATH, AS7343_CHANNELS, FWHM_std_factor
from cropspectai.data_analysis.preprocessing import split_dataset_into_X_y
import numpy as np
import pandas as pd


# def downsample_hyperspectra(X):

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

    for channel in AS7343_CHANNELS:

        reflectance[channel] = downsample_hyperspectra_to_channel(
            spectra,
            wavelengths,
            srf_df[channel].to_numpy(),
        )

    return pd.DataFrame(reflectance, index=features.index)

def generate_gaussian_srfs(wavelengths):

    srfs = {}

    for channel, params in AS7343_CHANNELS.items():
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
    sigma = fwhm / FWHM_std_factor 
    gaussian_density_values = np.exp(-0.5 * ((wavelengths - peak) / sigma) ** 2)

    return gaussian_density_values
    

if __name__ == "__main__":
    train_path = RAW_PARQUET_DATA_SPLITS_PATH / "train.parquet"
    # test_path = RAW_PARQUET_DATA_SPLITS_PATH / "test.parquet"

    df_train = pd.read_parquet(train_path)
    # df_test = pd.read_parquet(test_path)

    X_train, y_train = split_dataset_into_X_y(df_train)

    wavelengths = X_train.columns.astype(int).to_numpy()
    srf_df = generate_gaussian_srfs(wavelengths)
    print(f"SRF dataframe of train features: \n {srf_df}")
    # print(srf_df.describe())

    reflectance_df = downsample_to_as7343(X_train, wavelengths, srf_df)
    print(f"\nDownsampled reflectance of train features: \n {reflectance_df}")
