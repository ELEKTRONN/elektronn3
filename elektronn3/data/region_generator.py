# ELEKTRONN3 - Neural Network Toolkit
#
# Copyright (c) 2017 - now
# Max Planck Institute of Neurobiology, Munich, Germany
# Authors: Ravil Dorozhinskii

import numpy as np


class Region:
    """ Is a container that holds spatial coordinates
    of the region
    """
    def __init__(self,
                 coords_lo: list,
                 coords_hi: list,
                 size: list = None):
        """
        Parameters
        ----------
        coords_lo - the lowest region coordinates along each axis
        coords_hi - the highest region coordinates along each axis
        size - spatial sizes of the region along each axis
        """

        self.coords_lo = coords_lo
        self.coords_hi = coords_hi

        if size:
            self.size = size
        else:
            self.size = [high - low for high, low in zip(coords_hi, coords_lo)]


class RegionGenerator:
    """ A class instance generates regions with arbitrary
    spatial size and location within the specified coordinate bounds.
    The coordinate bounds are usually the spatial size of
    the input sample.
    """
    def __init__(self,
                 coord_bounds: list,
                 lower_lim_region_size: list,
                 upper_lim_region_size: list):
        """
        Parameters
        ----------
        coord_bounds - coordinate bounds of a sample
            with the format: [depth, width, height]
        lower_lim_region_size - region minimal size along each axis
            with the format: [min_depth, min_width, min_height]
        upper_lim_region_size - region maximal size along each axis
            with the format: [max_depth, max_width, max_height]
        """

        self.sample_size = coord_bounds
        self.coords_lo_lim = lower_lim_region_size
        self.coords_hi_lim = upper_lim_region_size
        self.dim = len(self.sample_size)

    def create_region(self) -> Region:
        """ Generates a region with arbitrary spatial size
        and location according to the parameters passed by the user
        to the constructor
        Returns
        -------
        instance of the Region class
        """

        size = [np.random.randint(low=self.coords_lo_lim[i],
                                  high=self.coords_hi_lim[i])
                for i in range(self.dim)]

        coords_lo = [np.random.randint(low=0,
                                       high=self.sample_size[i] - size[i])
                     for i in range(self.dim)]

        coords_hi = [coords_lo[i] + size[i] for i in range(self.dim)]

        return Region(coords_lo, coords_hi, size)
