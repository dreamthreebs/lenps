import numpy as np
import healpy as hp


def getObjectPixelNumbers_healpy(SNMap, threshold):
    """
    Detects connected regions in a HEALPix map above a given threshold and returns
    object IDs, maximum pixel indices, pixel counts, and a segmentation map.

    Parameters:
      SNMap (np.ndarray): 1D HEALPix map (length = 12*nside^2)
      threshold (float): threshold level for pixel selection

    Returns:
      objIDs (np.ndarray): array of object labels (starting from 1)
      objMaxPix (np.ndarray): array with the maximum S/N pixel index for each detected object
      objNumPix (np.ndarray): number of pixels in each object
      segmtMap (np.ndarray): segmentation map with object labels (0 for background)
    """
    # Determine the nside parameter from the total number of HEALPix pixels.
    nside = hp.npix2nside(len(SNMap))

    # Identify pixels with values above the given threshold and get their indices.
    sigPix_indices = np.nonzero(SNMap > threshold)[0]
    # Convert the list of significant pixel indices into a set for quick membership tests.
    sigPixSet = set(sigPix_indices)

    # Initialize a union-find (disjoint set) structure; each significant pixel is its own parent.
    parent = {pix: pix for pix in sigPixSet}

    def find(i):
        """
        Recursively find the representative (root) of the set to which pixel i belongs.
        This function also compresses the path for efficiency.
        """
        if parent[i] != i:
            parent[i] = find(parent[i])
        return parent[i]

    def union(i, j):
        """
        Union two sets by connecting the root of pixel j's set to the root of pixel i's set.
        This effectively joins two connected pixels.
        """
        root_i = find(i)
        root_j = find(j)
        if root_i != root_j:
            parent[root_j] = root_i

    # For every significant pixel, check its neighbors and join with those that are also above threshold.
    for pix in sigPixSet:
        # hp.get_all_neighbours returns the 8 neighbors (some entries might be -1 for invalid neighbors)
        neighbors = hp.get_all_neighbours(nside, pix)
        for nb in neighbors:
            if nb >= 0 and nb in sigPixSet:
                union(pix, nb)

    # Group all significant pixels by their connected components using the union-find results.
    components = {}
    for pix in sigPixSet:
        root = find(pix)
        # Append the pixel to the list corresponding to its component (root).
        components.setdefault(root, []).append(pix)

    # Assign new labels (starting from 1) for each connected component.
    component_labels = {root: i + 1 for i, root in enumerate(components.keys())}

    # Create the segmentation map where each pixel gets a label;
    # non-significant pixels (below the threshold) are set to 0.
    segmtMap = np.zeros_like(SNMap, dtype=int)
    for root, pixels in components.items():
        segmtMap[np.array(pixels)] = component_labels[root]

    # Prepare lists to hold the output data.
    objIDs = []  # Object labels
    objMaxPix = []  # Pixel numbers corresponding to the maximum value in the object
    objNumPix = []  # Number of pixels in the object

    # For each connected component, determine its label, total pixel count,
    # and the pixel number where SNMap is maximum.
    for root, pixels in components.items():
        label = component_labels[root]
        objIDs.append(label)
        objNumPix.append(len(pixels))
        # Identify the pixel in this component with the maximum value.
        max_pix = max(pixels, key=lambda x: SNMap[x])
        objMaxPix.append(max_pix)

    # Return the arrays for object IDs, maximum pixel numbers, pixel counts, and the segmentation map.
    return np.array(objIDs), np.array(objMaxPix), np.array(objNumPix), segmtMap
