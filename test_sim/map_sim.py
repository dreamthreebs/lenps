from pixell import enmap, utils, enplot
from astropy.coordinates import SkyCoord

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import astropy.units as u


def interactive_show(
    m: enmap.enmap,
    title="Map",
    unit="µK",
    cmap="coolwarm",
    vmin=None,
    vmax=None,
    plt_show=False,
):
    decmin, ramin = m.box()[0] / utils.degree
    decmax, ramax = m.box()[1] / utils.degree
    extent = [ramin, ramax, decmin, decmax]

    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(m, origin="lower", cmap=cmap, extent=extent, vmin=vmin, vmax=vmax)
    cbar = plt.colorbar(im, ax=ax, orientation="horizontal", pad=0.1, fraction=0.046)
    cbar.set_label(unit)

    ax.set_title(title)
    ax.set_xlabel("RA [deg]")
    ax.set_ylabel("Dec [deg]")

    def format_coord(x, y):
        pix = m.sky2pix([y * utils.degree, x * utils.degree])
        # pix = m.sky2pix([y, x])  # [Dec, RA]
        iy, ix = int(pix[0]), int(pix[1])
        if 0 <= iy < m.shape[-2] and 0 <= ix < m.shape[-1]:
            val = m[iy, ix]
            return f"RA={x:.3f}°, Dec={y:.3f}°, Value={val:.3f} {unit}"
        else:
            return f"RA={x:.3f}°, Dec={y:.3f}° (out of bounds)"

    # ax.format_coord = format_coord
    if plt_show:
        plt.show()


def gen_geometry():
    # Define area of map using numpy
    # pixell wants the box in the following format:
    # [[dec_from, RA_from], [dec_to, RA_to]]
    # Note RA goes "from" left "to" right!
    box = np.array([[25, 250], [70, 150]]) * utils.degree

    # Define a map geometry
    # the width and height of each pixel will be .5 arcmin
    shape, wcs = enmap.geometry(pos=box, res=0.5 * utils.arcmin, proj="car")

    # Create an empty ndmap
    empty_map = enmap.zeros(shape, wcs=wcs)
    print(f"{np.ceil(enmap.modlmap(shape, wcs).max())}")

    return empty_map, shape, wcs


def gen_cmb(omap):
    cl = np.load("../../cmbutils/data/cmbcl_8k.npy").T[0]
    print(f"{cl.shape=}")
    cmbmap = enmap.rand_map(omap.shape, omap.wcs, cov=cl)

    # nl = len(cl)
    # l = np.arange(nl)
    # plt.semilogy(l, l * (l + 1) / 2 / np.pi * cl, label="TT")
    # plt.show()

    return cmbmap


# make a polarized noise ps function
def T_and_P_noise_ps(
    ell,
    white_level=30,
    noise_ps_scaling=-4,
    T_knee=3000,
    T_cap=300,
    P_knee=300,
    P_cap=100,
):
    """Get the temperature and polarization noise power spectra evaluated at ell.
    Follows this model:

    PS(ell) = white_level**2 * ((ell/knee)**scaling + 1), ell > cap
    PS(ell) = white_level**2 * ((cap/knee)**scaling + 1), ell <= cap

    Parameters
    ----------
    ell : (...) np.ndarray
      Angular scales.
    white_level : scalar
      Temperature white noise level in uK-arcmin. Polarization is this times
      sqrt(2).
    noise_ps_scaling : scalar
      Power-law scaling of the low-ell noise power spectra.
    T_knee : scalar
      Ell-knee of temperature power spectrum.
    T_cap : scalar
      Minimum ell at which the spectrum is capped.
    P_knee : scalar
      Ell-knee of polarization power spectrum.
    P_cap : scalar
      Minimum ell at which the spectrum is capped.

    Returns
    -------
    (3, ...) np.ndarray
      The polarization noise power spectra in T, Q, U. Assumed diagonal over
      polarization.
    """
    T = np.zeros_like(ell)
    mask = ell <= T_cap
    T[mask] = white_level**2 * ((T_cap / T_knee) ** noise_ps_scaling + 1)
    T[~mask] = white_level**2 * ((ell[~mask] / T_knee) ** noise_ps_scaling + 1)

    P = np.zeros_like(ell)
    mask = ell <= P_cap
    P[mask] = 2 * white_level**2 * ((P_cap / P_knee) ** noise_ps_scaling + 1)
    P[~mask] = 2 * white_level**2 * ((ell[~mask] / P_knee) ** noise_ps_scaling + 1)

    # convert to steradians (note, not square radians!). the below lines first
    # convert to square radians, then from square radians to steradians
    T *= utils.arcmin**2 * (4 * np.pi / (np.pi * 2 * np.pi))
    P *= utils.arcmin**2 * (4 * np.pi / (np.pi * 2 * np.pi))

    # put into square shape and return
    out = np.zeros((3, 3, *ell.shape), ell.dtype)
    out[0, 0] = T
    out[1, 1] = P
    out[2, 2] = P

    return out


def get_noise_sim(shape, wcs, seed=None, **T_and_P_noise_ps_kwargs):
    """Draw a noise realization from the constructed noise powre spectrum
    that also accounts for the smaller, and thus noisier, pixels near the
    poles. this is curvedsky case!

    Parameters
    ----------
    shape : (ny, nx) tuple
      The footprint of the map.
    wcs : astropy.wcs.wcs.WCS
      The geometry of the map. Assumes units of degrees (the pixell
      default for wcs).
    seed : int or list of int
      Random seed.
    T_and_P_noise_ps_kwargs : dict
      Keyword arguments to be passed to T_and_P_noise_ps.

    Returns
    -------
    (3, ny, nx) enmap.ndmap
      Noise realization (polarized) drawn from T_and_P_noise_ps, with the
      correct white-noise level, correlated noise shape, and corrected for
      pixel areas.
    """
    # get the noise ps. for enmap.rand_map to work, this needs to be
    # evaluated at integer ells and cover all the angular scales in
    # out 2d fourier space
    ell = np.arange(0, np.ceil(enmap.modlmap(shape, wcs).max()) + 1)
    noise_ps = T_and_P_noise_ps(
        ell, white_level=10, noise_ps_scaling=0, **T_and_P_noise_ps_kwargs
    )

    # draw a noise realization
    noise_sim = enmap.rand_map((3, *shape), wcs, cov=noise_ps, seed=seed, scalar=True)

    # normalize by pixel area. we want this in terms of fraction of a
    # "flat-sky" pixel
    #
    pixsize_steradians = enmap.pixsizemap(shape, wcs, broadcastable=True)
    pix_area_deg = np.abs(np.prod(wcs.wcs.cdelt))
    pix_area_steradians = pix_area_deg * utils.degree**2
    frac_pixsize = pixsize_steradians / pix_area_steradians

    return noise_sim / np.sqrt(frac_pixsize)


def get_white_noise_map(shape, wcs, noise_uK_arcmin=10.0, seed=None):
    """
    直接在 map 空间生成 T/Q/U 噪声图像，满足给定的 μK·arcmin 水平。

    Parameters
    ----------
    shape : (ny, nx)
    wcs : pixell-compatible WCS
    noise_uK_arcmin : float
        白噪声强度，单位 μK·arcmin
    seed : int or None
        随机种子

    Returns
    -------
    enmap.ndmap of shape (3, ny, nx)
        T, Q, U 噪声图像
    """
    if seed is not None:
        np.random.seed(seed)

    # 获取 pixel 分辨率（单位：arcmin）
    res_arcmin = np.abs(wcs.wcs.cdelt[0]) * 60  # degrees → arcmin

    # 每个 pixel 的标准差
    std_per_pixel = noise_uK_arcmin / res_arcmin  # μK

    # 生成白噪声 map
    shape3 = (3,) + tuple(shape)
    noise = std_per_pixel * np.random.randn(*shape3)

    return enmap.enmap(noise, wcs)


def check_noise_cl():
    # do not need now!
    ell = np.arange(10000, dtype=np.float64)
    res_arcmin = np.abs(wcs.wcs.cdelt[0]) * 60
    print(f"{res_arcmin=}")
    noise_ps = T_and_P_noise_ps(ell, white_level=10, noise_ps_scaling=0)

    plt.loglog(ell, noise_ps[0, 0])
    plt.loglog(ell, noise_ps[1, 1])
    plt.show()


def gen_nemo_csv():
    df = pd.read_csv("95.csv")

    coords_gal = SkyCoord(
        l=df["lon"].values * u.rad, b=df["lat"].values * u.rad, frame="galactic"
    )
    coords = coords_gal.icrs

    df["RADeg"] = coords.ra.deg
    df["decDeg"] = coords.dec.deg

    # 创建 patch 的 enmap geometry（RA 150°–250°, Dec 25°–70°）
    box = np.deg2rad([[25, 250], [70, 150]])  # [[dec_min, ra_min], [dec_max, ra_max]]
    shape, wcs = enmap.geometry(pos=box, res=0.5 * utils.arcmin, proj="car")

    # 判断哪些点源在 patch 内
    pix_coords = np.array([coords.dec.rad, coords.ra.rad])
    inside_mask = enmap.contains(shape, wcs, pix_coords)

    # 保留下来的点源
    nemo_df = df[inside_mask].copy()

    # 保留 Nemo 需要的字段，并重命名 amplitude
    nemo_df = nemo_df[["index", "RADeg", "decDeg", "iflux"]].rename(
        columns={"iflux": "amplitude"}
    )

    # 写出 CSV，可直接被 Nemo catalogFileName 使用（单位：mJy）
    nemo_df.to_csv("input_95.csv", index=False)


if __name__ == "__main__":
    empty_map, m_shape, wcs = gen_geometry()
    # interactive_show(empty_map, plt_show=True)

    # cmb_map = gen_cmb(omap=empty_map)
    # interactive_show(cmb_map, plt_show=True)

    # check_noise_cl()
    m_noise = get_white_noise_map(shape=m_shape, wcs=wcs, noise_uK_arcmin=10, seed=42)
    print(f"{np.std(m_noise[0])=}")

    # interactive_show(m_noise[0], plt_show=True, vmin=-20, vmax=20)
    #
    gen_nemo_csv()
