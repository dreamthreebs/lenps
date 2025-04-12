import numpy as np
import healpy as hp
import pandas as pd
import matplotlib.pyplot as plt
import pysm3
import pysm3.units as u
import pymaster as nmt

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


def gen_cmb(beam, seed=0):
    cl = np.load("./data/cmbcl_8k.npy")

    np.random.seed(seed=seed)
    cmb = hp.synfast(
        cl.T[0], nside=nside, fwhm=np.deg2rad(beam) / 60, lmax=3 * nside - 1
    )

    return cmb


def gen_fg(nside, freq, beam=None):
    sky = pysm3.Sky(nside=nside, preset_strings=["d1", "s1", "f1"])
    fg = sky.get_emission(freq * u.GHz)
    fg = fg.to(u.uK_CMB, equivalencies=u.cmb_equivalencies(freq * u.GHz))[0]
    if beam is not None:
        fg = hp.smoothing(fg, fwhm=np.deg2rad(beam) / 60)
    return fg


def coadd_map(
    sim_type: str,
    nside: int,
    beam=None,
    nstd: float = 1.0,
    noise_seed: int = 0,
    cmb_seed: int = 0,
    freq: float = 95.0,
):
    if sim_type == "p":
        ps = np.load("./data/ps/ps.npy")[0]
        return ps
    elif sim_type == "n":
        noise = gen_noise(nstd=nstd, nside=nside, seed=noise_seed)
        return noise
    elif sim_type == "f":
        fg = gen_fg(nside=nside, freq=freq, beam=beam)
        return fg
    elif sim_type == "c":
        cmb = gen_cmb(beam=beam, seed=cmb_seed)
        return cmb
    elif sim_type == "pn":
        ps = np.load("./data/ps/ps.npy")[0]
        noise = gen_noise(nstd=nstd, nside=nside, seed=noise_seed)
        return ps + noise
    elif sim_type == "cfn":
        noise = gen_noise(nstd=nstd, nside=nside, seed=noise_seed)
        cmb = gen_cmb(beam=beam, seed=cmb_seed)
        fg = gen_fg(nside=nside, freq=freq, beam=beam)
        return cmb + noise + fg
    elif sim_type == "pcn":
        ps = np.load("./data/ps/ps.npy")[0]
        noise = gen_noise(nstd=nstd, nside=nside, seed=noise_seed)
        cmb = gen_cmb(beam=beam, seed=cmb_seed)
        return ps + cmb + noise
    elif sim_type == "pcfn":
        ps = np.load("./data/ps/ps.npy")[0]
        noise = gen_noise(nstd=nstd, nside=nside, seed=noise_seed)
        cmb = gen_cmb(beam=beam, seed=cmb_seed)
        fg = gen_fg(nside=nside, freq=freq, beam=beam)
        return ps + cmb + noise + fg
    else:
        raise ValueError(f"Unknown sim_type: {sim_type}")


def check_sim_all():
    m_pcfn = coadd_map(sim_type="pcfn", nside=nside, beam=beam, freq=30)
    m_pcn = coadd_map(sim_type="pcn", nside=nside, beam=beam)
    m_pn = coadd_map(sim_type="pn", nside=nside)
    m_p = coadd_map(sim_type="p", nside=nside)
    m_n = coadd_map(sim_type="n", nside=nside)
    plot_hp(m_pcfn, proj_type="moll", title="pcfn")
    plot_hp(m_pcn, proj_type="moll", title="pcn")
    plot_hp(m_pn, proj_type="moll", title="pn")
    plot_hp(m_p, proj_type="moll", title="p")
    plot_hp(m_n, proj_type="moll", title="n")


def check_local():
    m_pcfn = coadd_map(sim_type="pcfn", nside=nside, beam=beam, nstd=1, freq=30)
    # m_pn = coadd_map(sim_type="pn", nside=nside, nstd=10)
    m_p = coadd_map(sim_type="p", nside=nside)
    df = pd.read_csv("./mask/30.csv")
    flux_idx = 136
    lon = np.rad2deg(df.at[flux_idx, "lon"])
    lat = np.rad2deg(df.at[flux_idx, "lat"])
    plot_hp(
        m_pcfn,
        proj_type="gnom",
        rot=[lon, lat, 0],
        norm="linear",
        xsize=200,
        title="pcfn",
    )
    plot_hp(
        m_p, proj_type="gnom", rot=[lon, lat, 0], norm="linear", xsize=200, title="p"
    )


def check_cl(mask):
    m_pcfn = coadd_map(sim_type="pcfn", nside=nside, nstd=10, beam=beam, freq=freq)
    m_pcn = coadd_map(sim_type="pcn", nside=nside, nstd=10, beam=beam)
    m_cfn = coadd_map(sim_type="cfn", nside=nside, nstd=10, beam=beam, freq=freq)
    m_pn = coadd_map(sim_type="pn", nside=nside, nstd=10)
    m_n = coadd_map(sim_type="n", nside=nside, nstd=10)
    m_p = coadd_map(sim_type="p", nside=nside)
    cl_pcfn = hp.anafast(m_pcfn * mask, pol=False)
    cl_pcn = hp.anafast(m_pcn * mask, pol=False)
    cl_cfn = hp.anafast(m_cfn * mask, pol=False)
    cl_p = hp.anafast(m_p * mask, pol=False)
    cl_pn = hp.anafast(m_pn * mask, pol=False)
    cl_n = hp.anafast(m_n * mask, pol=False)
    l = np.arange(cl_pn.shape[-1])
    print(f"{l=}")

    np.save(f"./data/cl_total.npy", cl_cfn)

    plt.loglog(l, cl_pcfn, label="cl_pcfn")
    plt.loglog(l, cl_pcn, label="cl_pcn")
    plt.loglog(l, cl_cfn, label="cl_cfn")
    plt.loglog(l, cl_pn, label="cl_pn")
    plt.loglog(l, cl_p, label="cl_p")
    plt.loglog(l, cl_n, label="cl_n")
    plt.legend()
    plt.show()


def get_apo_mask():
    mask = np.load("./mask/BINMASKG.npy")
    mask_out = hp.ud_grade(mask, nside_out=nside)
    mask_out[mask_out < 1] = 0
    mask_apo = nmt.mask_apodization(mask_in=mask_out, aposize=3, apotype="C1")

    # hp.orthview(mask_out, rot=[100, 50, 0])
    # hp.orthview(mask_apo, rot=[100, 50, 0])
    # plt.show()
    return mask_apo


if __name__ == "__main__":
    apo_mask = get_apo_mask()
    # check_sim_all()
    # check_local()
    check_cl(mask=apo_mask)
