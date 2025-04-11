import numpy as np
import healpy as hp
import pandas as pd
import matplotlib.pyplot as plt

from config import lmax, nside, freq, beam
from cmbutils.plot import plot_hp
from find_peak import getObjectPixelNumbers_healpy
from gen_sim import coadd_map, check_local

m_pn = coadd_map(sim_type="pn", nside=nside, nstd=1)
# m_pcfn = coadd_map(sim_type="pcfn", nside=nside, beam=beam, freq=30)
# m_n = coadd_map(sim_type="n", nside=nside, nstd=1)
# std_pn = np.std(m_pn)
# std_n = np.std(m_n)
# print(f"{std_pn=}")
# print(f"{std_n=}")
id_arr, pix_ps, num_ps, _ = getObjectPixelNumbers_healpy(m_pn, threshold=50)
print(f"{id_arr=}")
print(f"{pix_ps=}")
print(f"{num_ps=}")

lon, lat = hp.pix2ang(nside=nside, ipix=pix_ps, lonlat=True)
print(f"{lon.shape=}")

#
# for idx in id_arr:
#     # if idx < 130:
#         # continue
#     print(f"{idx=}")
#     plot_hp(m_pn, proj_type="gnom", xsize=30, rot=[lon[idx], lat[idx], 0])
