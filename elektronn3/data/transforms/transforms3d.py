# ELEKTRONN3 - Neural Network Toolkit
#
# Copyright (c) 2019 - now
# Max Planck Institute of Neurobiology, Munich, Germany
# Authors: Jonathan Klimesch

import numpy as np
from scipy.spatial.transform import Rotation as Rot


class Compose3d:
    """ Composes several transforms together. """

    def __init__(self, transforms: list):
        self.transforms = transforms

    def __call__(self, points):
        for t in self.transforms:
            points = t(points)
        return points


class RandomRotate3d:
    """ Rotates given points in 3d space by using scipy euler rotations. """

    def __init__(self, angle_range: tuple = (-180, 180)):
        self.angle_range = angle_range

    def __call__(self, points):
        assert points.shape[1] == 3
        angles = np.random.uniform(self.angle_range[0], self.angle_range[1], (1, 3))[0]
        r = Rot.from_euler('xyz', angles, degrees=True)
        return r.apply(points)


class Center3d:
    """ Moves the centroid of the given points to the origin of the coordinate system and thereby
    centers the points. """

    def __call__(self, points):
        assert points.shape[1] == 3
        centroid = np.mean(points, axis=0)
        c_points = points - centroid
        return c_points


class RandomVariation3d:
    """ Adds some 3d variation to the given points. """

    def __init__(self, limits: tuple = (-1, 1)):
        if limits[0] < limits[1]:
            self.limits = limits
        elif limits[0] > limits[1]:
            self.limits = (limits[1], limits[0])
        else:
            self.limits = (0, 0)

    def __call__(self, points):
        if self.limits == (0, 0):
            return points
        assert points.shape[1] == 3
        variation = np.random.random(points.shape) * (self.limits[1] - self.limits[0]) + self.limits[0]
        return points+variation
