from .chg_method import ChargeMethod

import numpy as np
from numba import jit
from math import erf


@jit(nopython=True, cache=True)
def sqeqa_calculate(set_of_molecules, params, num_ats_types):
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
        list_of_qa = set_of_molecules.ref_av_charges[index:new_index]
        vector -= np.dot(matrix, list_of_qa)
        vector += list_of_hardness * list_of_qa
        A_sqe = np.dot(T, np.dot(matrix, T.T))
        B_sqe = np.dot(T, vector)
        for i, bond_params_index in enumerate(molecule.bonds_ids):
            A_sqe[i, i] += params[bond_params_index]
        results[index: new_index] = np.dot(np.linalg.solve(A_sqe, B_sqe), T) + list_of_qa
        index = new_index
    return results


class SQEqa(ChargeMethod):

    @property
    def params_pattern(self):
        return {"atom": {"data": {},
                         "names": ["chi", "eta", "width"]},
                "bond": {"data": {},
                         "name": "hardness"},
                "metadata": {"method": "SQEqa"}}

    @property
    def params_bounds(self):
        params_bounds = [[0, 5] for _ in range(len(self.params_vals))]
        return np.array(params_bounds)

    def calculate(self, set_of_molecules):
        return sqeqa_calculate(set_of_molecules, self.params_vals, len(self.ats_types))

    # add to parameterization.py
    # if str(chg_method) == "SQEqa":
    #     def add_av_ref_charges(set_of_mols_a):
    #         average_charges_list = []
    #         for mol in set_of_mols_a.mols:
    #             average_charges_list.extend(molecules_average_charges[mol.name])
    #         set_of_mols_a.ref_av_charges = np.array(average_charges_list, dtype=np.float64)
    #     set_of_mols_plain_ba = create_set_of_mols(sdf_file, "plain-ba")
    #     add_chgs(set_of_mols_plain_ba, ref_chgs_file, "ref_chgs")
    #     average_charges = {}
    #     for at_type in set_of_mols_plain_ba.ats_types:
    #         average_charges[at_type] = np.mean(set_of_mols_plain_ba.ref_ats_types_chgs[at_type])
    #     molecules_average_charges = {}
    #     for molecule in set_of_mols_plain_ba.mols:
    #         data = []
    #         for atom in molecule.ats_srepr:
    #             data.append(average_charges[atom])
    #         molecules_average_charges[molecule.name] = data
    #     add_av_ref_charges(set_of_mols)
    #     add_av_ref_charges(set_of_mols_par)
    #     add_av_ref_charges(subset_of_mols)
    #     add_av_ref_charges(min_subset_of_mols)
    #     add_av_ref_charges(set_of_mols_val)
