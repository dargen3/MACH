from .chg_method import ChargeMethod

from numba import jit
import numpy as np


@jit(nopython=True, cache=True)
def qeq_dasgupta_huzinaga_calculate(set_of_molecules, params):
    results = np.empty(set_of_molecules.num_of_ats, dtype=np.float64)
    index = 0
    for molecule in set_of_molecules.mols:
        num_of_at = molecule.num_of_ats
        new_index = index + num_of_at
        matrix = np.empty((num_of_at + 1, num_of_at + 1), dtype=np.float64)
        matrixB = np.empty((num_of_at, num_of_at), dtype=np.float64)
        for i, par_index_i in enumerate(molecule.ats_ids):
            for j, par_index_j in enumerate(molecule.ats_ids):
                d = molecule.distance_matrix[i][j]
                matrixB[i,j] = 1/(0.5*params[par_index_i+1]*np.e**(0.4*d) + 0.5*params[par_index_j+1]*np.e**(0.4*d))

        vector = np.empty(num_of_at + 1, dtype=np.float64)
        matrix[:num_of_at, :num_of_at] = 1 / (molecule.distance_matrix + matrixB)
        matrix[num_of_at, :] = 1.0
        matrix[:, num_of_at] = 1.0
        matrix[num_of_at, num_of_at] = 0.0
        for x, par_index in enumerate(molecule.ats_ids):
            matrix[x, x] = params[par_index + 1]
            vector[x] = -params[par_index]
        vector[-1] = molecule.total_chg
        results[index: new_index] = np.linalg.solve(matrix, vector)[:-1]
        index = new_index
    return results


class QEq_Dasgupta_Huzinaga(ChargeMethod):

    @property
    def params_pattern(self):
        return {"atom": {"data": {},
                         "names": ["electronegativity", "hardness"]},
                "metadata": {"method": "QEq_Dasgupta_Huzinaga"}}

    @property
    def params_bounds(self):
        return np.array([(0, 5) for x in range(len(self.params_vals))])

    def calculate(self, set_of_molecules):
        return qeq_dasgupta_huzinaga_calculate(set_of_molecules, self.params_vals)
