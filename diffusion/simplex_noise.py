"""Classes and functions for creating simplex noise."""

from typing import Optional, Tuple

import numpy as np
import opensimplex
import torch
from numba import njit
from torch import Tensor

MAX_SEED = np.iinfo(np.int32).max
MIN_SEED = np.iinfo(np.int32).min


class SimplexNoise:
    """Class for generating simplex noise."""

    def __init__(self, **kwargs):
        """Initialize simplex noise generator."""
        self.kwargs = kwargs

    def __call__(self, x_like: Tensor) -> Tensor:
        """Create a simplex array."""
        return self._get_noise(x_like.shape, **self.kwargs).to(x_like.device)

    @staticmethod
    def _get_noise(shape: Tuple, **kwargs) -> Tensor:
        """Compute simplex noise for a given shape."""
        bs = shape[0]
        im_shape = shape[-2:]
        return torch.from_numpy(
            batch_octavate2d(tuple(im_shape), int(bs), **kwargs)[:, None, :, :]
        ).float()


@njit
def rnd1(x: int) -> np.ndarray:
    """Init rnd1."""
    return np.random.randint(MIN_SEED, MAX_SEED, size=x)


@njit
def rnd2(x: int) -> np.ndarray:
    """Init rnd2."""
    return np.arange(x)


# Pseudo independent per batch
def batch_octavate2d(
    shape: Tuple,
    bs: int,
    octaves: int = 1,
    persistence: float = 0.5,
    frequency: float = 1 / 32,
    amplitude: float = 1,
    lacunarity: float = 0.5,
    perms: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Generate simplex noise on a batch basis."""
    # Set up random seeds/permutations for noise-
    # RNG cannot be called inside compiled numb.
    rndf = rnd2
    if perms is None:
        rndf = rnd1
        seeds = rndf(octaves)
        perms = np.array([opensimplex.internals._init(seed) for seed in seeds])

    @njit(cache=True, parallel=True)
    def octavate(
        shape, bs, octaves, persistence, frequency, amplitude, lacunarity, perms
    ):
        """Execute jit-compiled simplex noise function."""
        # Initialize grid, base resolution is 1 * shape, in batch dim randomly sample
        # locations to reduce dependancy in z since seed is for all z same
        noise = np.zeros((bs, shape[0], shape[1]))

        x = np.arange(shape[0])
        y = np.arange(shape[1])
        z = rndf(bs)

        # Further decorr through x, y offset
        xr = rndf(octaves)
        yr = rndf(octaves)

        frequencies = np.ones(octaves) * frequency
        frequencies[1:] = frequency * np.cumprod(lacunarity * np.ones(octaves - 1))

        amplitudes = np.ones(octaves) * amplitude
        amplitudes[1:] = amplitude * np.cumprod(persistence * np.ones(octaves - 1))

        for i in range(octaves):
            noise += amplitudes[i] * opensimplex.internals._noise3a(
                (x + xr[i]) * frequencies[i],
                (y + yr[i]) * frequencies[i],
                z,
                perms[i][0],
                perms[i][1],
            )
        return noise

    return octavate(
        shape, bs, octaves, persistence, frequency, amplitude, lacunarity, perms
    )
