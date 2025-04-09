import numpy as np
import healpy as hp
import pandas as pd
import matplotlib.pyplot as plt

from pathlib import Path
from config import freq, lmax, beam, nside


def mJy_to_uKCMB(intensity_mJy, frequency_GHz):
    # Constants
    c = 2.99792458e8  # Speed of light in m/s
    h = 6.62607015e-34  # Planck constant in J*s
    k = 1.380649e-23  # Boltzmann constant in J/K
    T_CMB = 2.725  # CMB temperature in Kelvin

    frequency_Hz = frequency_GHz * 1e9  # Convert frequency to Hz from GHz

    x = (h * frequency_Hz) / (k * T_CMB)  # Calculate x = h*nu/(k*T)

    # Calculate the derivative of the Planck function with respect to temperature, dB/dT
    dBdT = (2.0 * h * frequency_Hz**3 / c**2 / T_CMB) * (
        x * np.exp(x) / (np.exp(x) - 1) ** 2
    )
    intensity_Jy = intensity_mJy * 1e-3  # Convert intensity from mJy to Jy
    intensity_W_m2_sr_Hz = intensity_Jy * 1e-26  # Convert Jy/sr to W/m^2/sr/Hz
    uK_CMB = (
        intensity_W_m2_sr_Hz / dBdT * 1e6
    )  # Convert to uK_CMB, taking the inverse of dB/dT
    return uK_CMB


# basic parameters
npix = hp.nside2npix(nside=nside)
nside2pixarea_factor = hp.nside2pixarea(nside=nside)
df = pd.read_csv(f"./mask/{freq}.csv")
print(f"{len(df)=}")

# empty map
delta_t_map = np.zeros((npix,))
delta_q_map = np.zeros((npix,))
delta_u_map = np.zeros((npix,))
print(f"{delta_t_map.shape=}")

# generate map
for flux_idx in range(len(df)):
    print(f"{flux_idx=}")
    lon = np.rad2deg(df.at[flux_idx, "lon"])
    lat = np.rad2deg(df.at[flux_idx, "lat"])
    pflux = (
        mJy_to_uKCMB(df.at[flux_idx, "pflux"], frequency_GHz=freq)
        / nside2pixarea_factor
    )
    qflux = (
        mJy_to_uKCMB(df.at[flux_idx, "qflux"], frequency_GHz=freq)
        / nside2pixarea_factor
    )
    uflux = (
        mJy_to_uKCMB(df.at[flux_idx, "uflux"], frequency_GHz=freq)
        / nside2pixarea_factor
    )

    iflux = (
        mJy_to_uKCMB(df.at[flux_idx, "iflux"], frequency_GHz=freq)
        / nside2pixarea_factor
    )

    print(f"{lon=}, {lat=}, {pflux=}, {qflux=}, {uflux=}")

    ps_pix_idx = hp.ang2pix(nside=nside, theta=lon, phi=lat, lonlat=True)
    print(f"{ps_pix_idx=}")

    delta_t_map[ps_pix_idx] = iflux
    delta_q_map[ps_pix_idx] = qflux
    delta_u_map[ps_pix_idx] = uflux

# hp.gnomview(delta_q_map, rot=[lon, lat, 0], reso=1)
# plt.show()

m_ps = hp.smoothing(
    map_in=[delta_t_map, delta_q_map, delta_u_map], fwhm=np.deg2rad(beam) / 60, pol=True
)
path_ps = Path("./data/ps")
path_ps.mkdir(exist_ok=True, parents=True)
np.save("./data/ps/ps.npy", m_ps)
