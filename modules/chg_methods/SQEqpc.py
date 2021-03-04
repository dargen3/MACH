from .chg_method import ChargeMethod

import numpy as np
from numba import jit
from math import erf


@jit(nopython=True, cache=True)
def sqeqpc_calculate(set_of_molecules, params):

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
        list_of_q0 = np.empty(num_of_at, dtype=np.float64)
        list_of_hardness = np.empty(num_of_at, dtype=np.float64)
        for i, (par_index_i, ba_ats_ids) in enumerate(zip(molecule.ats_ids, molecule.ba_ats_ids)):






            c_hardness = params[par_index_i + 1]
            for ba in ba_ats_ids:
                if ba >=0:
                    c_hardness += params[ba+6]

            matrix[i, i] = c_hardness
            list_of_hardness[i] = c_hardness

            # matrix[i, i] = params[par_index_i + 1]
            # list_of_hardness[i] = params[par_index_i + 1]




            c_electronegativity = params[par_index_i]
            for ba in ba_ats_ids:
                if ba >= 0:
                    c_electronegativity += params[ba + 5]
            vector[i] = -c_electronegativity

            # vector[i] = -params[par_index_i]






            c_q0 = 0
            for ba in ba_ats_ids:
                if ba >= 0:
                    c_q0 += params[ba + 4]
            list_of_q0[i] = params[par_index_i + 3] + np.sqrt(c_q0)

            # list_of_q0[i] = params[par_index_i + 3]






            for j, (par_index_j, distance) in enumerate(zip(molecule.ats_ids[i + 1:],
                                                            molecule.distance_matrix[i, i + 1:]),
                                                        i + 1):




                width1 = params[par_index_i + 2]
                width2 = params[par_index_j + 2]


                for ba in ba_ats_ids:
                    if ba >= 0:
                        width1 += params[ba + 7]
                for ba2 in molecule.ba_ats_ids[j]:
                    if ba2 >= 0:
                        width2 += params[ba2 + 7]





                d0 = np.sqrt(2 * width1 ** 2 + 2 * width2 ** 2)



                matrix[i, j] = matrix[j, i] = erf(distance / d0) / distance
        list_of_q0 -= (np.sum(list_of_q0) - molecule.total_chg) / len(list_of_q0)
        vector -= np.dot(matrix, list_of_q0)
        vector += list_of_hardness * list_of_q0

        A_sqe = np.dot(T, np.dot(matrix, T.T))
        B_sqe = np.dot(T, vector)
        for i, bond_params_index in enumerate(molecule.bonds_ids):
            A_sqe[i, i] += params[bond_params_index]
        results[index: new_index] = np.dot(np.linalg.solve(A_sqe, B_sqe), T) + list_of_q0
        index = new_index
    return results


class SQEqpc(ChargeMethod):

    @property
    def params_pattern(self):
        return {"atom": {"data": {},
                         # "names": ["chi", "eta", "width", "q0", "chic", "etac", "q0c"]},
                         "names": ["chi", "eta", "width", "q0", "chic", "q0c", "as", "ddd"]},
                "bond": {"data": {},
                         "name": "hardness"},
                "metadata": {"method": "SQEqpc"}}

    @property
    def params_bounds(self):
        params_bounds = [[0, 5] for _ in range(len(self.params_vals))]
        for x in range(len(self.ats_types)):
            params_bounds[x * 8 + 3] = [-1, 1]
        return np.array(params_bounds)

    def calculate(self, set_of_molecules):
        return sqeqpc_calculate(set_of_molecules, self.params_vals)