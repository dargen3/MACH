from .chg_method import ChargeMethod

from numba import jit
import numpy as np


@jit(nopython=True, cache=True)
def eem_calculate(set_of_molecules, params):
    results = np.empty(set_of_molecules.num_of_ats, dtype=np.float64)
    kappa = params[-1]
    index = 0
    for molecule in set_of_molecules.mols:
        num_of_at = molecule.num_of_ats
        new_index = index + num_of_at
        matrix = np.empty((num_of_at + 1, num_of_at + 1), dtype=np.float64)
        vector = np.empty(num_of_at + 1, dtype=np.float64)
        matrix[:num_of_at, :num_of_at] = kappa / molecule.distance_matrix
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


class EEM(ChargeMethod):

    @property
    def params_pattern(self):
        return {"atom": {"data": {},
                         "names": ["electronegativity", "hardness"]},
                "common": {"kappa": 1},
                "metadata": {"method": "EEM"}}

    @property
    def params_bounds(self):
        return np.array([(0, 5) for x in range(len(self.params_vals))])

    def calculate(self, set_of_molecules):
        return eem_calculate(set_of_molecules, self.params_vals)
