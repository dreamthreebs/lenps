import numpy as np
import healpy as hp
import matplotlib.pyplot as plt


def beam_model(norm, beam, theta):
    sigma = np.deg2rad(beam) / 60 / (np.sqrt(8 * np.log(2)))
    return norm / (2 * np.pi * sigma**2) * np.exp(-(theta**2) / (2 * sigma**2))


def beam2bl(lmax, fwhm):
    # fwhm in degree
    theta = np.linspace(0, 30 * fwhm, 100)
    btheta = beam_model(1, fwhm)
