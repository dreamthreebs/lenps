from healpy.rotator import angdist
import numpy as np
import healpy as hp
import pandas as pd
import matplotlib.pyplot as plt

from config import lmax, nside, freq, beam
from cmbutils.plot import plot_hp
from cmbutils.mf import MatchedFilter
from find_peak import getObjectPixelNumbers_healpy
from gen_sim import coadd_map, get_apo_mask

apo_mask = get_apo_mask()
# m_pn = coadd_map(sim_type="pn", nside=nside, nstd=0)
# m_p = coadd_map(sim_type="p", nside=nside)
m_pcfn = coadd_map(sim_type="pcfn", nside=nside, beam=beam, freq=30, nstd=10)
m_cfn = coadd_map(sim_type="cfn", nside=nside, beam=beam, freq=30, nstd=10)

# m_n = coadd_map(sim_type="n", nside=nside, nstd=1)
# std_pn = np.std(m_pn)
# std_n = np.std(m_n)
# print(f"{std_pn=}")
# print(f"{std_n=}")
cl_tot = np.load("./data/cl_total.npy")

obj_mf = MatchedFilter(nside=nside, lmax=lmax, beam=beam, cl_tot=cl_tot)
obs_out, tot_out, snr, sigma, _ = obj_mf.run_mf(m_obs=m_pcfn, m_tot=m_cfn)

id_arr, pix_ps, num_ps, _ = getObjectPixelNumbers_healpy(snr * apo_mask, threshold=5)
print(f"{id_arr=}")
print(f"{pix_ps=}")
print(f"{num_ps=}")

lon, lat = hp.pix2ang(nside=nside, ipix=pix_ps, lonlat=True)
print(f"{lon.shape=}")

#
# for idx in id_arr:
#     # if idx < 108:
#     # continue
#     print(f"{idx=}")
#     plot_hp(
#         m_pcfn,
#         proj_type="gnom",
#         xsize=200,
#         rot=[lon[idx - 1], lat[idx - 1], 0],
#         title="m_pcfn",
#     )
#     plot_hp(
#         snr,
#         proj_type="gnom",
#         xsize=200,
#         rot=[lon[idx - 1], lat[idx - 1], 0],
#         title="snr",
#     )


df = pd.read_csv("./mask/30.csv")
lon_cat = np.rad2deg(df.loc[:, "lon"]).to_numpy()
lat_cat = np.rad2deg(df.loc[:, "lat"]).to_numpy()
print(f"{lon_cat=}")
print(f"{lat_cat=}")


dir_cat = (lon_cat, lat_cat)
for idx in range(len(lon)):
    dir_obs = (lon[idx], lat[idx])
    ang_dists = np.rad2deg(hp.rotator.angdist(dir1=dir_obs, dir2=dir_cat, lonlat=True))
    tag = (ang_dists < (1.8 * 9 / 60)).any()
    tag = np.min(ang_dists)

    print(f"{tag=}")
    # print(f"{idx=}, {ang_dists=}")
