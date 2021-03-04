from .chg_method import ChargeMethod

from numba import jit
import numpy as np


@jit(nopython=True, cache=True)
def qeq_louwen_vogt_calculate(set_of_molecules, params, num_ats_types):
    hardnesses = np.empty((num_ats_types * 2, num_ats_types * 2), dtype=np.float64)
    for x in range(num_ats_types):
        for y in range(num_ats_types):
            hardnesses[x * 2][y * 2] = 1 / ((2 * np.sqrt(params[x * 2 + 1] * params[y * 2 + 1])) ** 3)

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
            matrix[i, i] = params[par_index_i + 1]
            vector[i] = -params[par_index_i]
            for j, (par_index_j, distance) in enumerate(zip(molecule.ats_ids[i + 1:],
                                                            molecule.distance_matrix[i, i + 1:]),
                                                        i + 1):
                matrix[i, j] = matrix[j, i] = (hardnesses[par_index_i, par_index_j] + distance ** 3) ** (-1 / 3)
        vector[-1] = molecule.total_chg
        results[index: new_index] = np.linalg.solve(matrix, vector)[:-1]
        index = new_index
    return results


class QEq_Louwen_Vogt(ChargeMethod):

    @property
    def params_pattern(self):
        return {"atom": {"data": {},
                         "names": ["electronegativity", "hardness"]},
                "metadata": {"method": "QEq_Louwen_Vogt"}}

    @property
    def params_bounds(self):
        return np.array([(0, 5) for x in range(len(self.params_vals))])

    def calculate(self, set_of_molecules):
        return qeq_louwen_vogt_calculate(set_of_molecules, self.params_vals, len(self.ats_types))
