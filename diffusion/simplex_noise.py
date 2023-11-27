from typing import Tuple

import opensimplex
import numpy as np
import torch
from torch import Tensor

from numba import njit

MAX_SEED = np.iinfo(np.int32).max
MIN_SEED = np.iinfo(np.int32).min


class SimplexNoise:
    def __init__(self, **kwargs):
        self.kwargs = kwargs

    def __call__(self, x_like: Tensor) -> Tensor:
        return self._get_noise(x_like.shape, **self.kwargs).to(x_like.device)

    @staticmethod
    def _get_noise(shape: Tuple, **kwargs) -> Tensor:
        bs = shape[0]
        im_shape = shape[-2:]
        return torch.from_numpy(
            batch_octavate2d(tuple(im_shape), int(bs), **kwargs)[:, None, :, :]
        ).float()

@njit
def rnd1(x):
    return np.random.randint(MIN_SEED, MAX_SEED, size=x)


@njit
def rnd2(x):
    return np.arange(x)


# Pseudo independent per batch
def batch_octavate2d(shape, bs, octaves=1, persistence=0.5, frequency=1 / 32,
                     amplitude=1, lacunarity=0.5, perms=None):
    # Set up random seeds/permutations for noise, rng cannot be called inside numba compiled
    rndf = rnd2
    if perms is None:
        rndf = rnd1
        seeds = rndf(octaves)
        perms = np.array([opensimplex.internals._init(seed) for seed in seeds])

    @njit(cache=True, parallel=True)
    def octavate(shape, bs, octaves, persistence, frequency, amplitude, lacunarity,
                 perms):
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
                (x + xr[i]) * frequencies[i], (y + yr[i]) * frequencies[i], z,
                perms[i][0], perms[i][1])
        return noise

    # with warnings.catch_warnings():
    # warnings.simplefilter("ignore")
    # print(octavate.parallel_diagnostics(level=4))
    return octavate(shape, bs, octaves, persistence, frequency, amplitude, lacunarity,
                    perms)


# Pseudo independant per batch
def test_batch_octavate2d(shape, bs, octaves=1, persistence=0.5, frequency=1 / 32,
                          amplitude=1, lacunarity=0.5, perms=None):
    # Set up random seeds/permutations for noise, rng cannot be called inside numba compiled
    rndf = rnd2
    if perms is None:
        rndf = rnd1
        seeds = rndf(octaves)
        perms = np.array([opensimplex.internals._init(seed) for seed in seeds])

    # Initialize grid, base resolution is 1 * shape, in batch dim randomly sample locations to reduce dependancy in z since seed is for all z same
    noise = np.zeros((bs, *shape))
    x = np.arange(shape[0])
    y = np.arange(shape[1])
    z = rndf(bs)

    # Further decorr through x, y offset
    xr = rndf(octaves)
    yr = rndf(octaves)

    @njit(cache=True)  # , parallel=True)
    def octavate(noise, x, y, z, xr, yr, octaves, persistence, frequency, amplitude,
                 lacunarity, perms):
        for i in range(octaves):
            noise += amplitude * opensimplex.internals._noise3a((x + xr[i]) * frequency,
                                                                (y + yr[i]) * frequency,
                                                                z, perms[i][0],
                                                                perms[i][1])
            frequency *= lacunarity
            amplitude *= persistence
        return noise

    return octavate(noise, x, y, z, xr, yr, octaves, persistence, frequency, amplitude,
                    lacunarity, perms)


# depr
def octavate2d(x, y, octaves=1, persistence=0.5, frequency=32, amplitude=1,
               lacunarity=0.5, newseed=True):
    noise = np.zeros((len(x), len(y)))
    for _ in range(octaves):
        if newseed:
            seed = np.random.randint(MIN_SEED, MAX_SEED)
            opensimplex.seed(seed)
        noise += amplitude * opensimplex.noise2array(x * frequency, y * frequency)
        frequency *= lacunarity
        amplitude *= persistence
    return noise


def octavate3d(x, y, z, octaves=1, persistence=0.5, frequency=32, amplitude=1,
               lacunarity=0.5, newseed=True, batchshuffle=True):
    noise = np.zeros((len(z), len(x), len(y)))
    for _ in range(octaves):
        if newseed:
            seed = np.random.randint(MIN_SEED, MAX_SEED)
            opensimplex.seed(seed)
        if batchshuffle:
            z = np.random.random(len(z))
        noise += amplitude * opensimplex.noise3array(x * frequency, y * frequency,
                                                     z * frequency)
        frequency *= lacunarity
        amplitude *= persistence
    return noise


# Truly independant per batch
def tbatch_octavate2d(shape, bs, octaves=1, persistence=0.5, frequency=1 / 32,
                      amplitude=1, lacunarity=0.5, newseed=True, batchshuffle=True):
    # Set up random seeds/permutations for noise, rng cannot be called inside numba compiled
    seeds = np.random.randint(MIN_SEED, MAX_SEED, size=octaves * bs)
    perms = [opensimplex.internals._init(seed) for seed in seeds]

    # Initialize grid, base resolution is 1 * shape, in batch dim randomly sample locations to reduce dependancy in z since seed is for all z same
    noise = np.zeros((bs, *shape))
    x = np.arange(shape[0])
    y = np.arange(shape[1])

    @njit(cache=True)
    def octavate(noise, x, y, octaves, persistence, frequency, amplitude, lacunarity,
                 perms):
        for i in range(octaves):
            for j in range(bs):
                noise[j] = opensimplex.internals._noise2a(x * frequency, y * frequency,
                                                          perms[i * octaves + j][0])
            noise += amplitude * noise
            frequency *= lacunarity
            amplitude *= persistence
        return noise

    return octavate(noise, x, y, octaves, persistence, frequency, amplitude, lacunarity,
                    perms)
