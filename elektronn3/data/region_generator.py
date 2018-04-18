# ELEKTRONN3 - Neural Network Toolkit
#
# Copyright (c) 2017 - now
# Max Planck Institute of Neurobiology, Munich, Germany
# Authors: Martin Drawitsch, Philipp Schubert, Marius Killinger

import numpy as np

class RegionGenerator:
    """ A class instance generates blobs with arbitrary
    spatial size and location within the specified coordinate bounds.
    The the size coordinate bounds is usually the spatial size of
    the input sample.
    """
    def __init__(self, coord_bounds, lim_depth, lim_width, lim_height):
        """
        Parameters
        ----------
        coord_bounds - np.ndarray of int
            with the format: [ depth, width, height ]
        lim_depth - np.ndarray of int
            with the format: [min_depth, max_depth]
        lim_width - np.ndarray of int
            with the format: [min_width, max_width]
        lim_height - numpy array of integers
            with the format: [min_height, max_height]
        """
        self.sample_depth = coord_bounds[0]
        self.sample_width = coord_bounds[1]
        self.sample_height = coord_bounds[2]

        self.depth_min = lim_depth[0]
        self.depth_max = lim_depth[1]

        self.width_min = lim_width[0]
        self.width_max = lim_width[1]

        self.height_min = lim_height[0]
        self.height_max = lim_height[1]

    def create_region(self):
        """ Generates a blob with arbitrary spatial size
        and location according to the parameters passed by the user
        to the constructor
        Returns
        -------
        Blob - instance of Blob class
        """

        depth = np.random.randint(low=self.depth_min,
                                  high=self.depth_max)

        width = np.random.randint(low=self.width_min,
                                  high=self.width_max)

        height = np.random.randint(low=self.height_min,
                                   high=self.height_max)

        z_min = np.random.randint(low=0,
                                  high=self.sample_depth - depth)

        x_min = np.random.randint(low=0,
                                  high=self.sample_width - width)

        y_min = np.random.randint(low=0,
                                  high=self.sample_height - height)

        z_max = z_min + depth

        x_max = x_min + width

        y_max = y_min + height

        return Region(z_min, z_max, x_min, x_max, y_min, y_max)


class Region:
    """ Is a container that holds spatial coordinates
    of the blob
    """
    def __init__(self, z_min, z_max, x_min, x_max, y_min, y_max):
        """
        Parameters
        ----------
        z_min - int
        z_max - int
        x_min - int
        x_max - int
        y_min - int
        y_max - int
        """
        self.z_min = z_min
        self.z_max = z_max

        self.x_min = x_min
        self.x_max = x_max

        self.y_min = y_min
        self.y_max = y_max

        self.depth = z_max - z_min
        self.width = x_max - x_min
        self.height = y_max - y_min

    def __str__(self):
        return "z_min = %i; z_max = %i; x_min = %i;" \
               "x_max = %i; y_min = %i; y_max = %i; " \
               "depth = %i; width = %i; height = %i" \
               % (self.z_min, self.z_max,self.x_min,
                  self.x_max, self.y_min, self.y_max,
                  self.depth, self.width, self.height)
