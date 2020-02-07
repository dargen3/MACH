from numba import jit
from numpy import float32, linspace, zeros, random, arange


@jit(nopython=True, cache=True)
def lhs(dimensionality, samples, high_bound, low_bound):
    """ Latin hypercube sampling """
    cut = linspace(0, 1, samples + 1).astype(float32)
    u = zeros((samples, dimensionality), dtype=float32)
    for x in range(samples):
        u[x] += random.rand(dimensionality)
    for j in range(dimensionality):
        u[random.permutation(arange(samples)), j] = u[:, j] * cut[1] + cut[:samples]
    for x in range(samples):
        u[x] = u[x] * (high_bound - low_bound) + low_bound
    return u
