from .chg_method import ChargeMethod

import numpy as np
from numba import jit


@jit(nopython=True, cache=True)
def sqedasguptahuzinaga_calculate(set_of_molecules, params):
    results = np.empty(set_of_molecules.num_of_ats, dtype=np.float64)
    index = 0
    for molecule in set_of_molecules.mols:
        num_of_at = molecule.num_of_ats
        new_index = index + num_of_at
        T = np.zeros((len(molecule.bonds), num_of_at))
        bonds = molecule.bonds
        for i in range(len(bonds)):
            at1, at2, _ = bonds[i]
            T[i, at1] += 1
            T[i, at2] -= 1
        matrix = np.zeros((num_of_at, num_of_at))
        vector = np.zeros(num_of_at)
        for i, par_index_i in enumerate(molecule.ats_ids):
            matrix[i, i] = params[par_index_i + 1]
            vector[i] = -params[par_index_i]
            for j, (par_index_j, distance) in enumerate(zip(molecule.ats_ids[i + 1:],
                                                            molecule.distance_matrix[i, i + 1:]),
                                                        i + 1):
                d = molecule.distance_matrix[i][j]
                matrix[i, j] = matrix[j, i] = 1 / (d + 1/(0.5*params[par_index_i+1]*np.e**(0.4*d) + 0.5*params[par_index_j+1]*np.e**(0.4*d)))
        A_sqe = np.dot(T, np.dot(matrix, T.T))
        B_sqe = np.dot(T, vector)
        for i, bond_params_index in enumerate(molecule.bonds_ids):
            A_sqe[i, i] += params[bond_params_index]
        results[index: new_index] = np.dot(np.linalg.solve(A_sqe, B_sqe), T)
        index = new_index
    return results


class SQE_Dasgupta_Huzinaga(ChargeMethod):

    @property
    def params_pattern(self):
        return {"atom": {"data": {},
                         "names": ["chi", "eta"]},
                "bond": {"data": {},
                         "name": "hardness"},
                "metadata": {"method": "SQE_Dasgupta_Huzinaga"}}

    @property
    def params_bounds(self):
        params_bounds = [[-0, 5] for _ in range(len(self.params_vals))]
        return np.array(params_bounds)

    def calculate(self, set_of_molecules):
        return sqedasguptahuzinaga_calculate(set_of_molecules, self.params_vals)
