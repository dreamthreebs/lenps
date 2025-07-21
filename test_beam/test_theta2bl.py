import numpy as np
import healpy as hp
import matplotlib.pyplot as plt

from theta2bl import beam2map, beta2map, beta2bl, beam2bl


def ps_by_smoothing():
    nside = 2048
    npix = hp.nside2npix(nside=nside)
    m = np.zeros(shape=npix)
    print(f"{m.shape=}")
    ctr_val = 1
    ctr_pix = hp.ang2pix(nside=nside, theta=0, phi=0, lonlat=True)
    m[ctr_pix] = ctr_val
    m_ps = hp.smoothing(m, fwhm=np.deg2rad(9) / 60)
    m_ps = m_ps / np.max(m_ps)

    # hp.gnomview(m_ps)
    # plt.show()
    #
    return m_ps


def ps_by_btheta():
    nside = 2048
    m_ps = beam2map(nside=nside, fwhm=9, factor=5)
    # hp.gnomview(m_ps)
    # plt.show()
    return m_ps


def cpr_sm_btheta():
    m_ps_sm = ps_by_smoothing()
    m_ps_btheta = ps_by_btheta()
    hp.gnomview(m_ps_btheta - m_ps_sm)
    plt.show()


def sz_by_smoothing():
    nside = 2048
    lmax = 3 * nside - 1
    npix = hp.nside2npix(nside=nside)
    m = np.zeros(shape=npix)
    ctr_val = 1
    ctr_pix = hp.ang2pix(nside=nside, theta=0, phi=0, lonlat=True)
    m[ctr_pix] = ctr_val

    bl_sz = beta2bl(lmax=lmax, theta_ac=1, beta=2 / 3)
    bl_beam = beam2bl(lmax=lmax, fwhm=3)
    m_sz = hp.smoothing(m, beam_window=bl_sz * bl_beam)
    m_sz = m_sz / np.max(m_sz)
    # hp.gnomview(m_sz)
    # plt.show()

    return m_sz


def sz_by_beta():
    nside = 2048
    m_tsz = beta2map(nside=nside, theta_c=1, beta=2 / 3, factor=50)
    m_sz = hp.smoothing(m_tsz, fwhm=np.deg2rad(3) / 60)
    m_sz = m_sz / np.max(m_sz)

    # hp.gnomview(m_sz)
    # plt.show()
    #
    return m_sz


def cpr_tsz_sm_btheta():
    m_tsz_sm = sz_by_smoothing()
    m_tsz_btheta = sz_by_beta()
    hp.gnomview(m_tsz_sm, title="sm")
    hp.gnomview(m_tsz_btheta, title="beta")
    hp.gnomview(m_tsz_sm - m_tsz_btheta, title="diff")
    plt.show()


# ps_by_smoothing()
# ps_by_btheta()
# cpr_sm_btheta()

# sz_by_smoothing()
# sz_by_beta()
cpr_tsz_sm_btheta()
