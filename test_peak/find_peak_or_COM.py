import numpy as np
import healpy as hp


def getObjectLocations_healpy(SNMap, threshold, find_center_of_mass=False):
    """
    Detects connected regions in a HEALPix map above `threshold` and returns
    their IDs, longitudes, latitudes, pixel counts, and a segmentation map.

    Parameters
    ----------
    SNMap : np.ndarray
        1D HEALPix map (length = 12 * nside^2).
    threshold : float
        Threshold level for pixel selection.
    find_center_of_mass : bool, optional
        If False (default), use each object's peak-pixel location.
        If True, compute the S/N-weighted centre-of-mass on the sphere.

    Returns
    -------
    objIDs : np.ndarray, shape (N,)
        Labels 1…N for each detected object.
    objLon : np.ndarray, shape (N,)
        Longitudes (deg) of each object.
    objLat : np.ndarray, shape (N,)
        Latitudes (deg) of each object.
    objNumPix : np.ndarray, shape (N,)
        Pixel counts per object.
    segmtMap : np.ndarray, same shape as SNMap
        Segmentation map (0 = background, else object label).
    """
    # infer nside
    nside = hp.npix2nside(SNMap.size)

    # find all pixels above threshold
    sig_pix = np.nonzero(SNMap > threshold)[0]
    sig_set = set(sig_pix)

    # union-find init
    parent = {pix: pix for pix in sig_set}

    def find(i):
        if parent[i] != i:
            parent[i] = find(parent[i])
        return parent[i]

    def union(i, j):
        ri, rj = find(i), find(j)
        if ri != rj:
            parent[rj] = ri

    # connect neighbours
    for pix in sig_set:
        for nb in hp.get_all_neighbours(nside, pix):
            if nb >= 0 and nb in sig_set:
                union(pix, nb)

    # group into components
    comps = {}
    for pix in sig_set:
        root = find(pix)
        comps.setdefault(root, []).append(pix)

    # label them
    comp_label = {root: idx + 1 for idx, root in enumerate(comps)}
    segmtMap = np.zeros_like(SNMap, dtype=int)
    for root, pix_list in comps.items():
        segmtMap[pix_list] = comp_label[root]

    # prepare results
    objIDs = []
    objLon = []
    objLat = []
    objNumPix = []

    for root, pixels in comps.items():
        label = comp_label[root]
        objIDs.append(label)
        objNumPix.append(len(pixels))

        if not find_center_of_mass:
            # peak pixel → lon/lat
            peak = max(pixels, key=lambda p: SNMap[p])
            theta, phi = hp.pix2ang(nside, peak)
        else:
            # weighted COM in Cartesian → back to spherical
            masses = SNMap[pixels]
            theta_arr, phi_arr = hp.pix2ang(nside, pixels)
            x = np.sin(theta_arr) * np.cos(phi_arr)
            y = np.sin(theta_arr) * np.sin(phi_arr)
            z = np.cos(theta_arr)

            wsum = masses.sum()
            xw = (masses * x).sum() / wsum
            yw = (masses * y).sum() / wsum
            zw = (masses * z).sum() / wsum

            r = np.sqrt(xw * xw + yw * yw + zw * zw)
            xw, yw, zw = xw / r, yw / r, zw / r

            theta = np.arccos(zw)
            phi = np.arctan2(yw, xw)

        # convert to degrees
        lon = np.degrees(phi) % 360.0
        lat = 90.0 - np.degrees(theta)

        objLon.append(lon)
        objLat.append(lat)

    return (
        np.array(objIDs, dtype=int),
        np.array(objLon, dtype=float),
        np.array(objLat, dtype=float),
        np.array(objNumPix, dtype=int),
        segmtMap,
    )
