# ELEKTRONN3 - Neural Network Toolkit
#
# Copyright (c) 2017 - now
# Max Planck Institute of Neurobiology, Munich, Germany
# Authors: Martin Drawitsch, Philipp Schubert, Marius Killinger

import numpy as np

class BlobGenerator:
    """ A class instance generates blobs with arbitrary
    spacial size and location within specified domain.
    The domain size is usually the special size of the batch.
    The user is responsible to pass correct parameters.

    """
    def __init__(self, domain, lim_depth, lim_width, lim_height):
        """
        The construcure initializes all necessary class attributes
        and prepare an instance to generate blobs
        Parameters
        ----------
        domain - numpy array of integers
            with the format: [ depth, width, height ]
        lim_depth - numpy array of integers
            with the format: [min_depth, max_depth]
        lim_width - numpy array of integers
            with the format: [min_width, max_width]
        lim_height - numpy array of integers
            with the format: [min_height, max_height]
        """
        self.domain_depth = domain[0]
        self.domain_width = domain[1]
        self.domain_height = domain[2]

        self.depth_min = lim_depth[0]
        self.depth_max = lim_depth[1]

        self.width_min = lim_width[0]
        self.width_max = lim_width[1]

        self.height_min = lim_height[0]
        self.height_max = lim_height[1]

    def create_blob(self):
        """
        The function generates a blob with arbitrary spacial size
        and location according to the parameters passed by the user
        to the constructor
        Returns
        -------
        Blob - instance of Blob class
        """

        depth = np.random.randint(low=self.depth_min,
                                  high=self.depth_max,
                                  dtype=np.uint32)

        width = np.random.randint(low=self.width_min,
                                  high=self.width_max,
                                  dtype=np.uint32)

        height = np.random.randint(low=self.height_min,
                                   high=self.height_max,
                                   dtype=np.uint32)

        z_min = np.random.randint(low=0,
                                  high=self.domain_depth - depth,
                                  dtype=np.uint32)

        x_min = np.random.randint(low=0,
                                  high=self.domain_width - width,
                                  dtype=np.uint32)

        y_min = np.random.randint(low=0,
                                  high=self.domain_height - height,
                                  dtype=np.uint32)

        z_max = z_min + depth

        x_max = x_min + width

        y_max = y_min + height

        return Blob(z_min, z_max, x_min, x_max, y_min, y_max)


class Blob:
    """
    The class is a container that holds spacial coordinates
    of the blob
    """
    def __init__(self, z_min, z_max, x_min, x_max, y_min, y_max):
        """
        The constructor initializes the attributes and
        computes the depth, width and height of a blob instance
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
