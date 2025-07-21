import numpy as np
import healpy as hp
import matplotlib.pyplot as plt

from theta2bl import beam2map


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


# ps_by_smoothing()
# ps_by_btheta()
cpr_sm_btheta()
