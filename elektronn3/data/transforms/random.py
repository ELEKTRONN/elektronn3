"""Random number generators for random augmentation parametrization"""

from typing import Optional, Tuple

import numpy as np
import scipy.stats


class RandomSampler:
    """Samples random variables from a ``scipy.stats`` distribution."""
    def __init__(
            self,
            rv: scipy.stats.rv_continuous,
            shape: Tuple[int, ...] = (),
            bounds: Optional[Tuple[float, float]] = None,
    ):
        self.rv = rv
        self.shape = shape
        self.bounds = bounds

    def __call__(self, shape=None):
        shape = self.shape if shape is None else shape
        rand = self.rv.rvs(size=shape)
        if self.bounds is not None:
            lo, hi = self.bounds
            rand = np.clip(rand, lo, hi)
        return rand


class Normal(RandomSampler):
    """Normal distribution sampler."""
    def __init__(
            self,
            mean: float = 0,
            sigma: float = 1,
            shape: Tuple[int, ...] = (),
            bounds: Optional[Tuple[float, float]] = None,
    ):
        rv = scipy.stats.norm(loc=mean, scale=sigma)
        super().__init__(rv=rv, shape=shape, bounds=bounds)


class HalfNormal(RandomSampler):
    """Half-normal distribution sampler.

    See https://en.wikipedia.org/wiki/Half-normal_distribution.
    Note that all sampled values are positive, regardless of the parameters."""
    def __init__(
            self,
            sigma: float = 1,
            shape: Tuple[int, ...] = (),
            bounds: Optional[Tuple[float, float]] = None,
    ):
        rv = scipy.stats.halfnorm(loc=0, scale=sigma)
        super().__init__(rv=rv, shape=shape, bounds=bounds)


class RandInt(RandomSampler):
    """Discrete uniform distribution sampler

    Outputs random integers in a defined range ``(low, high)`` with equal
    probability.

    By default (``low=0, high=2``), it generates binary values (0 or 1)."""
    def __init__(
            self,
            low: int = 0,
            high: int = 2,
            shape: Tuple[int, ...] = (),
    ):
        rv = scipy.stats.randint(low=low, high=high)
        super().__init__(rv=rv, shape=shape, bounds=None)
