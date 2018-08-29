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
            rng: Optional[np.random.RandomState] = None
    ):
        self.rv = rv
        self.shape = shape
        self.bounds = bounds
        self.rng = np.random.RandomState() if rng is None else rng

    def __call__(self, shape=None):
        shape = self.shape if shape is None else shape
        rand = self.rv.rvs(size=shape, random_state=self.rng)
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
            rng: Optional[np.random.RandomState] = None
    ):
        rv = scipy.stats.norm(loc=mean, scale=sigma)
        super().__init__(rv=rv, shape=shape, bounds=bounds, rng=rng)


class HalfNormal(RandomSampler):
    """Half-normal distribution sampler.

    See https://en.wikipedia.org/wiki/Half-normal_distribution.
    Note that all sampled values are positive, regardless of the parameters."""
    def __init__(
            self,
            sigma: float = 1,
            shape: Tuple[int, ...] = (),
            bounds: Optional[Tuple[float, float]] = None,
            rng: Optional[np.random.RandomState] = None
    ):
        rv = scipy.stats.halfnorm(loc=0, scale=sigma)
        super().__init__(rv=rv, shape=shape, bounds=bounds, rng=rng)


class RandInt(RandomSampler):
    """RandInt distribution sampler. Default is binary"""
    def __init__(
            self,
            low: int = 0,
            high: int = 2,
            shape: Tuple[int, ...] = (),
            rng: Optional[np.random.RandomState] = None
    ):
        rv = scipy.stats.randint(low=low, high=high)
        super().__init__(rv=rv, shape=shape, rng=rng, bounds=None)
