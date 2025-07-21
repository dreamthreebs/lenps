import numpy as np
import healpy as hp
import matplotlib.pyplot as plt

from gen_sim import gen_ksz

nside = 512
beam = None


for freq in [90, 150, 217, 270]:
    m_tsz = gen_ksz(nside=nside, freq=freq, beam=beam)
    hp.mollview(m_tsz, title=f"{freq=}", norm="hist")
    plt.show()
