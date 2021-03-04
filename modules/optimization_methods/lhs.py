import numpy as np
from numba import jit


@jit(nopython=True, cache=True)
def lhs(num_samples: int,
        bounds: np.array) -> np.array:

    dimensionality = len(bounds)
    cut = np.linspace(0, 1, num_samples + 1).astype(np.float32)
    samples = np.zeros((num_samples, dimensionality), dtype=np.float32)
    for x in range(num_samples):
        samples[x] += np.random.rand(dimensionality)
    for j in range(dimensionality):
        samples[np.random.permutation(np.arange(num_samples)), j] = samples[:, j] * cut[1] + cut[:num_samples]
    for x in range(num_samples):
        for y, (low_bound, high_bound) in enumerate(bounds):
            samples[x][y] = samples[x][y] * (high_bound - low_bound) + low_bound
    return samples
