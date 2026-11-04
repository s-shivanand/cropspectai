# Model input features

[Project overview](../README.md) · [App guide](../app/README.md)

**19 inputs = 12 reflectance channels + 7 engineered features**

The tables follow the current `v_01` model’s [configuration](../models/xgb_best_parameters_v_01.json).
Equations match the [code](../ml/src/cropspectai/data_analysis/feature_engineering.py)
and use AS7343 band approximations.

## Reflectance channels

Channel values are calibrated reflectance. The symbols identify the channels
used in the equations below; rows follow the model’s input order.

| Channel | Wavelength | Equation symbol |
| --- | --- | --- |
| `F1` | 405 nm | — |
| `F2` | 425 nm | — |
| `FZ` | 450 nm | $B$ |
| `F3` | 475 nm | — |
| `F4` | 515 nm | $S$ |
| `FY` | 555 nm | — |
| `F5` | 550 nm | $G$ |
| `FXL` | 600 nm | — |
| `F6` | 640 nm | — |
| `F7` | 690 nm | $R$ |
| `F8` | 745 nm | $E$ |
| `NIR` | 855 nm | $N$ |

## Engineered features

These features are calculated from the averaged reflectance spectrum and passed
to the model in the order shown.

| Feature | Description | Equation |
| --- | --- | --- |
| `mndvi_as7343` | Modified NDVI | $\frac{N-R}{N+R-2B}$ |
| `sipi_as7343` | Structure Insensitive Pigment Index | $\frac{N-B}{N-R}$ |
| `ci_red_edge_as7343` | Red-edge chlorophyll index | $\frac{N}{E}-1$ |
| `psri_as7343` | Plant Senescence Reflectance Index | $\frac{R-S}{E}$ |
| `red_green_nd` | Red–green normalized difference | $\frac{R-G}{R+G}$ |
| `datt1_as7343` | Datt difference ratio | $\frac{N-E}{N-R}$ |
| `mcari_osavi_as7343` | MCARI/OSAVI ratio | $\frac{\mathrm{MCARI}}{\mathrm{OSAVI}}$ |

The final feature uses:

$$
\mathrm{MCARI}=[(E-R)-0.2(E-G)]\frac{E}{R}
$$

$$
\mathrm{OSAVI}=1.16\frac{N-R}{N+R+0.16}
$$
