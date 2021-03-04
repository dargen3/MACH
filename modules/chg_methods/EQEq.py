from .chg_method import ChargeMethod

from numba import jit
import numpy as np


@jit(nopython=True, cache=True)
def eqeq_calculate(set_of_molecules, params, num_ats_types):
    k_parameter = params[-2]
    lambda_parameter = params[-1]
    J_vals = np.empty(num_ats_types * 2, dtype=np.float64)
    X_vals = np.empty(num_ats_types * 2, dtype=np.float64)
    for x in range(num_ats_types):
        J_vals[x * 2] = params[x * 2 + 1] - params[x * 2]
        X_vals[x * 2] = (params[x * 2 + 1] + params[x * 2]) / 2
    a_vals = np.empty((num_ats_types * 2, num_ats_types * 2), dtype=np.float64)
    for x in range(num_ats_types):
        for y in range(num_ats_types):
            a_vals[x * 2][y * 2] = np.sqrt(J_vals[x * 2] * J_vals[y * 2]) / k_parameter
    results = np.empty(set_of_molecules.num_of_ats, dtype=np.float64)
    index = 0
    for molecule in set_of_molecules.mols:
        num_of_at = molecule.num_of_ats
        new_index = index + num_of_at
        matrix = np.empty((num_of_at + 1, num_of_at + 1), dtype=np.float64)
        vector = np.empty(num_of_at + 1, dtype=np.float64)
        matrix[num_of_at, :] = 1.0
        matrix[:, num_of_at] = 1.0
        matrix[num_of_at, num_of_at] = 0.0
        for i, par_index_i in enumerate(molecule.ats_ids):
            matrix[i, i] = J_vals[par_index_i]
            vector[i] = -X_vals[par_index_i]
            for j, (par_index_j, distance) in enumerate(zip(molecule.ats_ids[i + 1:],
                                                            molecule.distance_matrix[i, i + 1:]),
                                                        i + 1):
                a = a_vals[par_index_i, par_index_j]
                overlap = np.exp(-a * a * distance ** 2) * (2 * a - a * a * distance - 1 / distance)
                matrix[i, j] = matrix[j, i] = lambda_parameter * k_parameter / 2 * (1 / distance + overlap)
        vector[-1] = molecule.total_chg
        results[index: new_index] = np.linalg.solve(matrix, vector)[:-1]
        index = new_index
    return results


class EQEq(ChargeMethod):

    @property
    def params_pattern(self):
        return {"atom": {"data": {},
                         "names": ["electron_affinity", "ionization_potencial"]},
                "common": {"lambda": 1, "k": 1},
                "metadata": {"method": "EQEq"}}

    @property
    def params_bounds(self):
        params_bounds = []
        for _ in range(int((len(self.params_vals)-2)/2)):
            params_bounds.append([0.0, 1.0])
            params_bounds.append([1.0, 5.0])
        params_bounds.append([0.0, 5.0])
        params_bounds.append([0.0, 5.0])
        return np.array(params_bounds)

    def calculate(self, set_of_molecules):
        return eqeq_calculate(set_of_molecules, self.params_vals, len(self.ats_types))
