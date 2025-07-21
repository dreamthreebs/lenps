import numpy as np
import healpy as hp
import matplotlib.pyplot as plt


def beam_model(norm_beam, theta, FWHM):
    """
    Gaussian beam model.
    """
    sigma = np.deg2rad(FWHM) / 60 / (np.sqrt(8 * np.log(2)))
    return norm_beam / (2 * np.pi * sigma**2) * np.exp(-((theta) ** 2) / (2 * sigma**2))


def beta_model(norm_beam, theta, theta_c, beta):
    """
    Beta model profile for the tSZ effect.
    """
    temp = (1 + theta**2 / np.deg2rad(theta_c / 60) ** 2) ** (-(3 * beta - 1) / 2)
    return norm_beam * temp


def beam2bl(lmax, fwhm):
    """
    Convert beam(theta) to b(l).
    """
    # fwhm = np.deg2rad(fwhm/60)  # arcmin
    # sigma = fwhm / (np.sqrt(8 * np.log(2)))
    theta = np.linspace(0, 2 * fwhm, 30000)
    btheta = beam_model(1, theta, fwhm)
    b_ell = hp.beam2bl(btheta, theta, lmax=lmax)
    b_ell /= b_ell[0]  # normalize
    return b_ell


def beta2bl(lmax, theta_ac, beta=1):
    """
    Convert Compton-y(theta) to b(l).
    """
    # theta_ac = np.deg2rad(theta_ac/60)  # arcmin
    theta = np.linspace(0, 2 * theta_ac, 30000)
    btheta = beta_model(1, theta, theta_ac, beta=beta)
    b_ell = hp.beam2bl(btheta, theta, lmax=lmax)
    b_ell /= b_ell[0]  # normalize
    return b_ell


def beam2map(
    nside,
    fwhm,
    factor=2,
    theta=0,
    phi=0,
):
    """
    Create a Gaussian beam map with a given FWHM and center coordinates.
    """
    from astropy.coordinates import SkyCoord
    import astropy.units as u

    radius_factor = factor
    FWHM = fwhm  # arcmin
    radius_deg = radius_factor * FWHM / 60  # degrees
    radius_rad = np.radians(radius_deg)  # radians

    # 创建一个全空的HEALPix map
    n_pix = hp.nside2npix(nside)  # 计算像素数量
    ps_map = np.zeros(n_pix)  # 初始化为0的掩模图（值为0表示非源区域）

    centre_pix = hp.ang2pix(nside=nside, theta=theta, phi=phi, lonlat=True)
    centre_vec = np.array(hp.pix2vec(nside=nside, ipix=centre_pix)).astype(np.float64)

    ipix_fit = hp.query_disc(nside=nside, vec=centre_vec, radius=radius_rad)
    vec_around = np.array(hp.pix2vec(nside=nside, ipix=ipix_fit.astype(int))).astype(
        np.float64
    )

    cos_theta = centre_vec @ vec_around
    cos_theta = np.clip(cos_theta, -1, 1)
    theta = np.arccos(cos_theta)

    ps_map[ipix_fit] = beam_model(1, theta, FWHM)

    ps_map /= np.max(ps_map)  # Normalize the map to 1
    return ps_map


def beta2map(nside, theta_c, beta=1, factor=2, ra=0, dec=90, coord="C"):
    """
    Create a beta model map with a given core radius and center coordinates.
    """
    from astropy.coordinates import SkyCoord
    import astropy.units as u

    radius_factor = factor
    theta_c = theta_c  # arcmin
    radius_deg = radius_factor * theta_c / 60  # degrees
    radius_rad = np.radians(radius_deg)  # radians

    # 创建一个全空的HEALPix map
    n_pix = hp.nside2npix(nside)  # 计算像素数量
    ps_map = np.zeros(n_pix)  # 初始化为0的掩模图（值为0表示非源区域）

    if coord == "G":
        source = SkyCoord(l=ra * u.deg, b=dec * u.deg, frame="galactic")
        # HEALPix 使用 theta（天顶角，0°在北极）和 phi（经度）
        theta = np.deg2rad(90 - source.b.deg)  # colatitude = 90 - dec
        phi = np.deg2rad(source.l.deg)
    elif coord == "C":
        source = SkyCoord(ra=ra * u.deg, dec=dec * u.deg, frame="icrs")
        # HEALPix 使用 theta（天顶角，0°在北极）和 phi（经度）
        theta = np.deg2rad(90 - source.dec.deg)  # colatitude = 90 - dec
        phi = np.deg2rad(source.ra.deg)

    centre_pix = hp.ang2pix(
        nside=nside,
        theta=theta,
        phi=phi,
    )
    centre_vec = np.array(hp.pix2vec(nside=nside, ipix=centre_pix)).astype(np.float64)

    ipix_fit = hp.query_disc(nside=nside, vec=centre_vec, radius=radius_rad)
    vec_around = np.array(hp.pix2vec(nside=nside, ipix=ipix_fit.astype(int))).astype(
        np.float64
    )

    centre_vec = np.asarray(
        hp.ang2vec(
            theta=theta,
            phi=phi,
        )
    )
    cos_theta = centre_vec @ vec_around
    cos_theta = np.clip(cos_theta, -1, 1)
    theta = np.arccos(cos_theta)

    ps_map[ipix_fit] = beta_model(1, theta, theta_c, beta)
    ps_map /= np.max(ps_map)  # Normalize the map to 1
    return ps_map
