from .chg_method import ChargeMethod

import numpy as np
from numba import jit
from math import erf


@jit(nopython=True, cache=True)
def sqeq0_calculate(set_of_molecules, params, num_ats_types):
    multiplied_widths = np.empty((num_ats_types * 3, num_ats_types * 3), dtype=np.float64)
    for x in range(num_ats_types):
        for y in range(num_ats_types):
            multiplied_widths[x * 3][y * 3] = np.sqrt(2 * params[x * 3 + 2] ** 2 + 2 * params[y * 3 + 2] ** 2)

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
        list_of_hardness = np.empty(num_of_at, dtype=np.float64)
        for i, par_index_i in enumerate(molecule.ats_ids):
            matrix[i, i] = params[par_index_i + 1]
            list_of_hardness[i] = params[par_index_i + 1]
            vector[i] = -params[par_index_i]
            for j, (par_index_j, distance) in enumerate(zip(molecule.ats_ids[i + 1:],
                                                            molecule.distance_matrix[i, i + 1:]),
                                                        i + 1):
                d0 = multiplied_widths[par_index_i, par_index_j]
                matrix[i, j] = matrix[j, i] = erf(distance / d0) / distance
        list_of_q0 = molecule.formal_chgs
        vector -= np.dot(matrix, list_of_q0)
        vector += list_of_hardness * list_of_q0
        A_sqe = np.dot(T, np.dot(matrix, T.T))
        B_sqe = np.dot(T, vector)
        for i, bond_params_index in enumerate(molecule.bonds_ids):
            A_sqe[i, i] += params[bond_params_index]
        results[index: new_index] = np.dot(np.linalg.solve(A_sqe, B_sqe), T) + list_of_q0
        index = new_index
    return results


class SQEq0(ChargeMethod):

    @property
    def params_pattern(self):
        return {"atom": {"data": {},
                         "names": ["chi", "eta", "width"]},
                "bond": {"data": {},
                         "name": "hardness"},
                "metadata": {"method": "SQEq0"}}

    @property
    def params_bounds(self):
        params_bounds = [[0, 5] for _ in range(len(self.params_vals))]
        return np.array(params_bounds)

    def calculate(self, set_of_molecules):
        return sqeq0_calculate(set_of_molecules, self.params_vals, len(self.ats_types))
