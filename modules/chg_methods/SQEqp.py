from .chg_method import ChargeMethod

import numpy as np
from numba import jit
from math import erf


@jit(nopython=True, cache=True)
def sqeqp_calculate(set_of_molecules, params, num_ats_types):
    multiplied_widths = np.empty((num_ats_types * 4, num_ats_types * 4), dtype=np.float64)
    for x in range(num_ats_types):
        for y in range(num_ats_types):
            multiplied_widths[x * 4][y * 4] = np.sqrt(2 * params[x * 4 + 2] ** 2 + 2 * params[y * 4 + 2] ** 2)

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
        for i, par_index_i in enumerate(molecule.ats_ids):
            matrix[i, i] = params[par_index_i + 1]
            list_of_hardness[i] = params[par_index_i + 1]
            vector[i] = -params[par_index_i]
            list_of_q0[i] = params[par_index_i + 3]
            for j, (par_index_j, distance) in enumerate(zip(molecule.ats_ids[i + 1:],
                                                            molecule.distance_matrix[i, i + 1:]),
                                                        i + 1):
                d0 = multiplied_widths[par_index_i, par_index_j]
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


class SQEqp(ChargeMethod):

    @property
    def params_pattern(self):
        return {"atom": {"data": {},
                         "names": ["chi", "eta", "width", "q0"]},
                "bond": {"data": {},
                         "name": "hardness"},
                "metadata": {"method": "SQEqp"}}

    @property
    def params_bounds(self):
        params_bounds = [[0, 5] for _ in range(len(self.params_vals))]
        for x in range(len(self.ats_types)):
            params_bounds[x * 4 + 3] = [-1, 1]
        return np.array(params_bounds)

    def calculate(self, set_of_molecules):
        return sqeqp_calculate(set_of_molecules, self.params_vals, len(self.ats_types))


# @jit(nopython=True, cache=True)
# def sqeqpopt_calculate(set_of_molecules, params, num_ats_types):
#     multiplied_widths = np.empty((num_ats_types * 4, num_ats_types * 4), dtype=np.float64)
#     for x in range(num_ats_types):
#         for y in range(num_ats_types):
#             multiplied_widths[x * 4][y * 4] = np.sqrt(2 * params[x * 4 + 2] ** 2 + 2 * params[y * 4 + 2] ** 2)
#
#     results = np.empty(set_of_molecules.num_of_ats, dtype=np.float64)
#     index = 0
#     for molecule in set_of_molecules.mols:
#         num_of_at = molecule.num_of_ats
#         new_index = index + num_of_at
#         A_sqe = np.zeros((molecule.num_of_bonds, molecule.num_of_bonds))
#         list_of_q0 = np.empty(molecule.num_of_ats, dtype=np.float64)
#         B_sqe = np.zeros(molecule.num_of_bonds)
#         bonds = molecule.bonds
#         T = np.zeros((len(molecule.bonds), num_of_at))
#         for i in range(len(bonds)):
#             at1, at2, _ = bonds[i]
#             T[i, at1] += 1
#             T[i, at2] -= 1
#         matrix = np.zeros((num_of_at, num_of_at))
#         list_of_hardness = np.empty(num_of_at, dtype=np.float64)
#         for i, par_index_i in enumerate(molecule.ats_ids):
#             list_of_q0[i] = params[par_index_i + 3]
#             matrix[i, i] = params[par_index_i + 1]
#             list_of_hardness[i] = params[par_index_i + 1]
#             for j, (par_index_j, distance) in enumerate(zip(molecule.ats_ids[i + 1:],
#                                                             molecule.distance_matrix[i, i + 1:]),
#                                                         i + 1):
#                 d0 = multiplied_widths[par_index_i, par_index_j]
#                 matrix[i, j] = matrix[j, i] = erf(distance / d0) / distance
#         list_of_q0 -= (np.sum(list_of_q0) - molecule.total_chg) / len(list_of_q0)
#         for i, bond_params_index in enumerate(molecule.bonds_ids): # mozna prepsat! numba už to snad umi
#             at1, at2, _ = bonds[i]
#             at1_at2_overlap = matrix[at1][at2]
#             A_sqe[i, i] = -2*at1_at2_overlap + list_of_hardness[at1] + list_of_hardness[at2] + params[bond_params_index]
#             B_sqe[i] = params[molecule.ats_ids[at1]] - at1_at2_overlap * list_of_q0[at1] - params[molecule.ats_ids[at2]] + at1_at2_overlap * list_of_q0[at2]
#             for j, par_index_j in enumerate(molecule.ats_ids):
#                 if j in [at1, at2]:
#                     continue
#                 else:
#                     B_sqe[i] += matrix[at1][j]*list_of_q0[j] - matrix[at2][j]*list_of_q0[j]
#
#             for j in range(i+1, molecule.num_of_bonds):
#                 at2_1, at2_2, _ = bonds[j]
#                 if at1 == at2_1:
#                     A_sqe[i, j] = A_sqe[j, i] = -1*(matrix[at1][at2] + matrix[at2_1][at2_2] - list_of_hardness[at1] - matrix[at2][at2_2])
#
#                 elif at1 == at2_2:
#                     A_sqe[i, j] = A_sqe[j, i] = (matrix[at1][at2] + matrix[at2_1][at2_2] - list_of_hardness[at1] - matrix[at2][at2_1])
#
#                 elif at2 == at2_1:
#                     A_sqe[i, j] = A_sqe[j, i] = (matrix[at1][at2] + matrix[at2_1][at2_2] - list_of_hardness[at2] - matrix[at1][at2_2])
#
#                 elif at2 == at2_2:
#                     A_sqe[i, j] = A_sqe[j, i] = -1*(matrix[at1][at2] + matrix[at2_1][at2_2] - list_of_hardness[at2] - matrix[at1][at2_1])
#
#                 else:
#                     A_sqe[i, j] = A_sqe[j, i] = matrix[at1][at2_1] + matrix[at2][at2_2] - matrix[at2][at2_1] - matrix[at1][at2_2]
#         results[index: new_index] = np.dot(np.linalg.solve(A_sqe, -B_sqe), T) + list_of_q0
#         index = new_index
#     return results
#
#
# class SQEqpopt(ChargeMethod):
#
#     @property
#     def params_pattern(self):
#         return {"atom": {"data": {},
#                          "names": ["chi", "eta", "width"]},
#                 "bond": {"data": {},
#                          "name": "hardness"},
#                 "metadata": {"method": "SQEqpopt"}}
#
#     @property
#     def params_bounds(self):
#         params_bounds = [[-0, 5] for _ in range(len(self.params_vals))]
#         return np.array(params_bounds)
#
#     def calculate(self, set_of_molecules):
#
#         return sqeqpopt_calculate(set_of_molecules, self.params_vals, len(self.ats_types))
