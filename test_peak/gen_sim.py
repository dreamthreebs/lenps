import numpy as np
import healpy as hp
import pandas as pd
import matplotlib.pyplot as plt

from config import lmax, nside, freq, beam
from cmbutils.plot import plot_hp


def gen_noise(nstd: float, nside: int = 256, seed=None):
    """
    Generate Gaussian noise for a HEALPix map with mean 0 and specified standard deviation.

    Parameters:
    -----------
    nstd : float
        Standard deviation of the noise.
    nside : int, optional
        Resolution parameter for the HEALPix map (default: 256).
    seed : int, optional
        Seed for reproducibility (default: None).

    Returns:
    --------
    np.ndarray
        Noise map as a HEALPix array.
    """
    rng = np.random.default_rng(seed)  # Modern random number generator
    npix = hp.nside2npix(nside)
    noise = nstd * rng.standard_normal(
        npix
    )  # Gaussian noise with mean 0, std=1 (scaled by nstd)
    return noise


def coadd_map(sim_type: str, nside: int, nstd: float = 1.0, noise_seed: int = 0):
    if sim_type == "p":
        ps = np.load("./data/ps/ps.npy")[0]
        return ps
    elif sim_type == "pn":
        ps = np.load("./data/ps/ps.npy")[0]
        noise = gen_noise(nstd=nstd, nside=nside, seed=noise_seed)
        return ps + noise
    elif sim_type == "n":
        noise = gen_noise(nstd=nstd, nside=nside, seed=noise_seed)
        return noise
    else:
        raise ValueError(f"Unknown sim_type: {sim_type}")


def check_sim_all():
    m_pn = coadd_map(sim_type="pn", nside=nside)
    m_p = coadd_map(sim_type="p", nside=nside)
    m_n = coadd_map(sim_type="n", nside=nside)
    plot_hp(m_pn, proj_type="moll")
    plot_hp(m_p, proj_type="moll")
    plot_hp(m_n, proj_type="moll")


def check_local():
    m_pn = coadd_map(sim_type="pn", nside=nside, nstd=10)
    m_p = coadd_map(sim_type="p", nside=nside)
    df = pd.read_csv("./mask/30.csv")
    flux_idx = 136
    lon = np.rad2deg(df.at[flux_idx, "lon"])
    lat = np.rad2deg(df.at[flux_idx, "lat"])
    plot_hp(m_p, proj_type="gnom", rot=[lon, lat, 0], norm="linear", xsize=20)


def check_cl():
    m_pn = coadd_map(sim_type="pn", nside=nside, nstd=10)
    m_p = coadd_map(sim_type="p", nside=nside)
    m_n = coadd_map(sim_type="n", nside=nside, nstd=10)
    cl_pn = hp.anafast(m_pn, pol=False)
    cl_p = hp.anafast(m_p, pol=False)
    cl_n = hp.anafast(m_n, pol=False)
    print(f"{cl_pn.shape=}")
    print(f"{cl_p.shape=}")
    print(f"{cl_n.shape=}")
    l = np.arange(cl_pn.shape[-1])
    print(f"{l=}")

    plt.loglog(l, cl_pn, label="cl_pn")
    plt.loglog(l, cl_p, label="cl_p")
    plt.loglog(l, cl_n, label="cl_n")
    plt.legend()
    plt.show()


if __name__ == "__main__":
    check_local()
    # check_cl()
