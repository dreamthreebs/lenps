from pixell import enmap, utils, enplot
import numpy as np
import matplotlib.pyplot as plt

from pathlib import Path

# Define area of map using numpy
# pixell wants the box in the following format:
# [[dec_from, RA_from], [dec_to, RA_to]]
# Note RA goes "from" left "to" right!
box = np.array([[-5, 10], [5, -10]]) * utils.degree

# Define a map geometry
# the width and height of each pixel will be .5 arcmin
shape, wcs = enmap.geometry(pos=box, res=0.5 * utils.arcmin, proj="car")
print(f"{shape=}")

# Create an empty ndmap
empty_map = enmap.zeros((3,) + shape, wcs=wcs)

# Check out the ndmap
# does the shape make sense given the bounding box and resolution?
print(empty_map.shape)
print(empty_map.dtype)
print(empty_map + np.pi)
print(empty_map[0, 10:15, 90:95] == 0)

print(empty_map.wcs)

stacked_map = np.concatenate([empty_map, empty_map])
print(stacked_map.shape)

dec_min = -7
ra_min = 5
dec_max = 3
ra_max = -5

# All coordinates in pixell are specified in radians
box = np.array([[dec_min, ra_min], [dec_max, ra_max]]) * utils.degree

imap_box = enmap.read_map(
    "../../lenps_data/act_planck_dr5.01_s08s18_AA_f150_night_map_d56_I.fits", box=box
)

enplot.pshow((imap_box, "act_map.png"), colorbar=True, downgrade=2)
