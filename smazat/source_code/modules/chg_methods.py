from datetime import date
from json import load, dumps
from math import erf
from sys import exit

import numpy as np
from numba import jit
from termcolor import colored


class ChargeMethod:
    def __repr__(self):
        return self.__class__.__name__

    def load_params(self,
                    params_file: str,
                    ats_types_pattern: str = None) -> str:

        print(f"Loading of parameters from {params_file}...")
        if params_file:
            self.params = load(open(params_file))
            self.ats_types_pattern = self.params["metadata"]["atomic_types_pattern"]
        else:
            self.params = self.params_pattern
            self.ats_types_pattern = ats_types_pattern
            self.params["metadata"]["atomic_types_pattern"] = ats_types_pattern

        self.params_per_at_type = len(self.params["atom"]["names"])
        method_in_params_file = self.params["metadata"]["method"]
        if self.__class__.__name__ != method_in_params_file:
            exit(colored(f"ERROR! These parameters are for method {method_in_params_file}, "
                         f"but you selected by argument --chg_method {self.__class__.__name__}!\n", "red"))
        print(colored("ok\n", "green"))
        return self.ats_types_pattern

    def prepare_params_for_calc(self,
                                set_of_molecules: "SetOfMolecules"):
        missing_at = set(set_of_molecules.ats_types) - set(self.params["atom"]["data"].keys())
        exit_status = False
        if missing_at:
            print(colored(f"ERROR! Atomic type(s) {', '.join(missing_at)} is not defined in parameters.", "red"))
            exit_status = True
        if "bond" in self.params:
            missing_bonds = set(set_of_molecules.bonds_types) - set(self.params["bond"]["data"].keys())
            if missing_bonds:
                print(colored(f"ERROR! Bond type(s) {', '.join(missing_bonds)} is not defined in parameters.", "red"))
                exit_status = True
        if exit_status:
            exit()
        self._dict_to_array()

    def prepare_params_for_par(self,
                               set_of_molecules: "SetOfMolecules"):
        missing_at = set(set_of_molecules.ats_types) - set(self.params["atom"]["data"].keys())
        for at in missing_at:
            for key, vals in self.params["atom"]["data"].items():
                if key.split("/")[0] == at.split("/")[0]:
                    self.params["atom"]["data"][at] = vals
                    print(colored(f"    Atom type {at} was added to parameters. "
                                  f"Parameters derived from {key}", "yellow"))
                    break
            else:
                self.params["atom"]["data"][at] = [np.random.random() for _ in range(len(self.params["atom"]["names"]))]
                print(colored(f"    Atom type {at} was added to parameters. Parameters are random numbers.", "yellow"))
        # unused atoms are atomic types which are in parameters but not in set of molecules
        unused_at = set(self.params["atom"]["data"].keys()) - set(set_of_molecules.ats_types)
        for at in unused_at:
            del self.params["atom"]["data"][at]
        if unused_at:
            print(colored(f"    {', '.join(unused_at)} was deleted from parameters, "
                          f"because of absence in set of molecules.", "yellow"))

        if "bond" in self.params:
            missing_bonds = set(set_of_molecules.bonds_types) - set(self.params["bond"]["data"].keys())
            for bond in missing_bonds:
                mat1, mat2, mtype = [value.split("/")[0] for value in bond.split("-")]
                for key, val in self.params["bond"]["data"].items():
                    pat1, pat2, ptype = [value.split("/")[0] for value in key.split("-")]
                    if (pat1, pat2, ptype) == (mat1, mat2, mtype):
                        self.params["bond"]["data"][bond] = val
                        print(colored(f"    Bond type {bond} was added to parameters. "
                                      f"Parameter derived from {key}", "yellow"))
                        break
                else:
                    self.params["bond"]["data"][bond] = np.random.random()
                    print(colored(f"    Bond type {bond} was added to parameters. "
                                  f"Parameter is random numbers.", "yellow"))
            # unused bonds are bond types which are in parameters but not in set of molecules
            unused_bonds = set(self.params["bond"]["data"].keys()) - set(set_of_molecules.bonds_types)
            for bond in unused_bonds:
                del self.params["bond"]["data"][bond]
            if unused_bonds:
                print(colored(f"    {', '.join(unused_bonds)} was deleted from parameters, "
                              f"because of absence in set of molecules", "yellow"))
        self._dict_to_array()

    def _dict_to_array(self):
        self.ats_types = sorted(self.params["atom"]["data"].keys())
        if "bond" in self.params:
            self.bond_types = sorted(self.params["bond"]["data"].keys())
        params_vals = []
        for _, vals in sorted(self.params["atom"]["data"].items()):
            params_vals.extend(vals)
        if "bond" in self.params:
            for _, val in sorted(self.params["bond"]["data"].items()):
                params_vals.append(val)
        if "common" in self.params:
            params_vals.extend(self.params["common"].values())
        self.params_vals = np.array(params_vals, dtype=np.float64)
        self.bounds = (min(self.params_vals), max(self.params_vals))

    def new_params(self,
                   new_params: np.array,
                   sdf_file: str,
                   new_params_file: str,
                   original_params_file: str,
                   ref_chg_file: str,
                   date: date):

        print(f"Writing parameters to {new_params_file}...")
        self.params_vals = new_params
        params_per_at = len(self.params["atom"]["names"])
        index = 0
        for at in self.ats_types:
            self.params["atom"]["data"][at] = list(self.params_vals[index: index + params_per_at])
            index = index + params_per_at

        if "bond" in self.params:
            for bond in self.bond_types:
                self.params["bond"]["data"][bond] = self.params_vals[index]
                index += 1

        if "common" in self.params:
            for common in sorted(self.params["common"]):
                self.params["common"][common] = self.params_vals[index]
                index += 1

        self.params["metadata"]["sdf file"] = sdf_file
        self.params["metadata"]["reference charges file"] = ref_chg_file
        self.params["metadata"]["original parameters file"] = original_params_file
        self.params["metadata"]["date"] = date

        with open(new_params_file, "w") as new_params_file:
            new_params_file.write(dumps(self.params, indent=2, sort_keys=True))
        print(colored("ok\n", "green"))


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


@jit(nopython=True, cache=True)
def qeq_calculate(set_of_molecules, params, num_ats_types):
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


class QEq(ChargeMethod):

    @property
    def params_pattern(self):
        return {"atom": {"data": {},
                         "names": ["electronegativity", "hardness"]},
                "common": {"kappa": 1},
                "metadata": {"method": "QEq"}}

    @property
    def params_bounds(self):
        return np.array([(0, 5) for x in range(len(self.params_vals))])

    def calculate(self, set_of_molecules):
        return qeq_calculate(set_of_molecules, self.params_vals, len(self.ats_types))


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


@jit(nopython=True, cache=True)
def sqe_calculate(set_of_molecules, params, num_ats_types):
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
        # print(f"T:\n{T}\n\n\n")
        matrix = np.zeros((num_of_at, num_of_at))
        vector = np.zeros(num_of_at)
        for i, par_index_i in enumerate(molecule.ats_ids):
            matrix[i, i] = params[par_index_i + 1]
            vector[i] = -params[par_index_i]
            for j, (par_index_j, distance) in enumerate(zip(molecule.ats_ids[i + 1:],
                                                            molecule.distance_matrix[i, i + 1:]),
                                                        i + 1):
                d0 = multiplied_widths[par_index_i, par_index_j]
                matrix[i, j] = matrix[j, i] = erf(distance / d0) / distance
        # print(f"vector:\n{vector}\n\n\n")
        # 
        # print(f"matrix:\n{matrix}\n\n\n")

        A_sqe = np.dot(T, np.dot(matrix, T.T))

        # print(f"T.T: \n{T.T}\n\n\n")
        # print(f"np.dot(matrix,T.T): \n{np.dot(matrix, T.T)}\n\n\n")
        # print(f"A_sqe:\n{A_sqe}\n\n\n")

        B_sqe = np.dot(T, vector)
        # print(f"T:\n{T}\n\n\n")
        # print(f"vector:\n{vector}\n\n\n")
        # print(f"B_sqe:\n{B_sqe}\n\n\n")


        for i, bond_params_index in enumerate(molecule.bonds_ids):
            A_sqe[i, i] += params[bond_params_index]

        # np.set_printoptions(threshold=np.inf)
        # print(A_sqe)
        # print("")
        # print(np.dot(np.linalg.solve(A_sqe, B_sqe), T))
        # exit()


        results[index: new_index] = np.dot(np.linalg.solve(A_sqe, B_sqe), T)

        # print(f"np.linalg.solve(A_sqe, B_sqe):\n{np.linalg.solve(A_sqe, B_sqe)}\n\n\n")
        # print(f"np.dot(np.linalg.solve(A_sqe, B_sqe), T):\n{np.dot(np.linalg.solve(A_sqe, B_sqe), T)}\n\n\n")



        index = new_index
    return results


class SQE(ChargeMethod):

    @property
    def params_pattern(self):
        return {"atom": {"data": {},
                         "names": ["chi", "eta", "width"]},
                "bond": {"data": {},
                         "name": "hardness"},
                "metadata": {"method": "SQE"}}

    @property
    def params_bounds(self):
        params_bounds = [[-0, 5] for _ in range(len(self.params_vals))]
        return np.array(params_bounds)

    def calculate(self, set_of_molecules):

        return sqe_calculate(set_of_molecules, self.params_vals, len(self.ats_types))


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
        # print(f"prekryvy:\n{matrix}")
        #
        # print(f"vector(el):\n{vector}")
        # print(f"hardness:\n{list_of_hardness}")

        list_of_q0 -= (np.sum(list_of_q0) - molecule.total_chg) / len(list_of_q0)

        # print(f"q0:\n{list_of_q0}")
        vector -= np.dot(matrix, list_of_q0)

        # print(f"vector1:\n{vector}")


        vector += list_of_hardness * list_of_q0
        # print(f"vector2:\n{vector}\n")

        A_sqe = np.dot(T, np.dot(matrix, T.T))
        B_sqe = np.dot(T, vector)
        # print(T, vector) # smata
        # print(B_sqe)# smata
        # exit()# smata
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


#########################################¨¨¨¨¨¨¨¨









@jit(nopython=True, cache=True)
def sqeopt_calculate(set_of_molecules, params, num_ats_types):
    multiplied_widths = np.empty((num_ats_types * 3, num_ats_types * 3), dtype=np.float64)
    for x in range(num_ats_types):
        for y in range(num_ats_types):
            multiplied_widths[x * 3][y * 3] = np.sqrt(2 * params[x * 3 + 2] ** 2 + 2 * params[y * 3 + 2] ** 2)

    results = np.empty(set_of_molecules.num_of_ats, dtype=np.float64)
    index = 0
    for molecule in set_of_molecules.mols:
        num_of_at = molecule.num_of_ats
        new_index = index + num_of_at
        A_sqe = np.zeros((molecule.num_of_bonds, molecule.num_of_bonds))
        B_sqe = np.zeros(molecule.num_of_bonds)
        bonds = molecule.bonds
        T = np.zeros((len(molecule.bonds), num_of_at))
        for i in range(len(bonds)):
            at1, at2, _ = bonds[i]
            T[i, at1] += 1
            T[i, at2] -= 1
        matrix = np.zeros((num_of_at, num_of_at))
        list_of_hardness = np.empty(num_of_at, dtype=np.float64)
        for i, par_index_i in enumerate(molecule.ats_ids):
            matrix[i, i] = params[par_index_i + 1]
            list_of_hardness[i] = params[par_index_i + 1]
            for j, (par_index_j, distance) in enumerate(zip(molecule.ats_ids[i + 1:],
                                                            molecule.distance_matrix[i, i + 1:]),
                                                        i + 1):
                d0 = multiplied_widths[par_index_i, par_index_j]
                matrix[i, j] = matrix[j, i] = erf(distance / d0) / distance
        for i, bond_params_index in enumerate(molecule.bonds_ids): # mozna prepsat! numba už to snad umi
            at1, at2, _ = bonds[i]
            A_sqe[i, i] = -2*matrix[at1][at2] + list_of_hardness[at1] + list_of_hardness[at2] + params[bond_params_index]
            B_sqe[i] = -params[molecule.ats_ids[at1]] + params[molecule.ats_ids[at2]]
            for j in range(i+1, molecule.num_of_bonds):
                at2_1, at2_2, _ = bonds[j]
                if at1 == at2_1:
                    A_sqe[i, j] = A_sqe[j, i] = -1*(matrix[at1][at2] + matrix[at2_1][at2_2] - list_of_hardness[at1] - matrix[at2][at2_2])

                elif at1 == at2_2:
                    A_sqe[i, j] = A_sqe[j, i] = (matrix[at1][at2] + matrix[at2_1][at2_2] - list_of_hardness[at1] - matrix[at2][at2_1])

                elif at2 == at2_1:
                    A_sqe[i, j] = A_sqe[j, i] = (matrix[at1][at2] + matrix[at2_1][at2_2] - list_of_hardness[at2] - matrix[at1][at2_2])

                elif at2 == at2_2:
                    A_sqe[i, j] = A_sqe[j, i] = -1*(matrix[at1][at2] + matrix[at2_1][at2_2] - list_of_hardness[at2] - matrix[at1][at2_1])

                else:
                   A_sqe[i, j] = A_sqe[j, i] = matrix[at1][at2_1] + matrix[at2][at2_2] - matrix[at2][at2_1] - matrix[at1][at2_2]
        results[index: new_index] = np.dot(np.linalg.solve(A_sqe, B_sqe), T)
        index = new_index
    return results


class SQEopt(ChargeMethod):

    @property
    def params_pattern(self):
        return {"atom": {"data": {},
                         "names": ["chi", "eta", "width"]},
                "bond": {"data": {},
                         "name": "hardness"},
                "metadata": {"method": "SQEopt"}}

    @property
    def params_bounds(self):
        params_bounds = [[-0, 5] for _ in range(len(self.params_vals))]
        return np.array(params_bounds)

    def calculate(self, set_of_molecules):

        return sqeopt_calculate(set_of_molecules, self.params_vals, len(self.ats_types))




















@jit(nopython=True, cache=True)
def sqeqpopt_calculate(set_of_molecules, params, num_ats_types):
    multiplied_widths = np.empty((num_ats_types * 4, num_ats_types * 4), dtype=np.float64)
    for x in range(num_ats_types):
        for y in range(num_ats_types):
            multiplied_widths[x * 4][y * 4] = np.sqrt(2 * params[x * 4 + 2] ** 2 + 2 * params[y * 4 + 2] ** 2)

    results = np.empty(set_of_molecules.num_of_ats, dtype=np.float64)
    index = 0
    for molecule in set_of_molecules.mols:
        num_of_at = molecule.num_of_ats
        new_index = index + num_of_at
        A_sqe = np.zeros((molecule.num_of_bonds, molecule.num_of_bonds))
        list_of_q0 = np.empty(molecule.num_of_ats, dtype=np.float64)
        B_sqe = np.zeros(molecule.num_of_bonds)
        bonds = molecule.bonds
        T = np.zeros((len(molecule.bonds), num_of_at))
        for i in range(len(bonds)):
            at1, at2, _ = bonds[i]
            T[i, at1] += 1
            T[i, at2] -= 1
        matrix = np.zeros((num_of_at, num_of_at))
        list_of_hardness = np.empty(num_of_at, dtype=np.float64)
        for i, par_index_i in enumerate(molecule.ats_ids):
            list_of_q0[i] = params[par_index_i + 3]
            matrix[i, i] = params[par_index_i + 1]
            list_of_hardness[i] = params[par_index_i + 1]
            for j, (par_index_j, distance) in enumerate(zip(molecule.ats_ids[i + 1:],
                                                            molecule.distance_matrix[i, i + 1:]),
                                                        i + 1):
                d0 = multiplied_widths[par_index_i, par_index_j]
                matrix[i, j] = matrix[j, i] = erf(distance / d0) / distance
        list_of_q0 -= (np.sum(list_of_q0) - molecule.total_chg) / len(list_of_q0)
        for i, bond_params_index in enumerate(molecule.bonds_ids): # mozna prepsat! numba už to snad umi
            at1, at2, _ = bonds[i]
            at1_at2_overlap = matrix[at1][at2]
            A_sqe[i, i] = -2*at1_at2_overlap + list_of_hardness[at1] + list_of_hardness[at2] + params[bond_params_index]
            B_sqe[i] = params[molecule.ats_ids[at1]] - at1_at2_overlap * list_of_q0[at1] - params[molecule.ats_ids[at2]] + at1_at2_overlap * list_of_q0[at2]
            for j, par_index_j in enumerate(molecule.ats_ids):
                if j in [at1, at2]:
                    continue
                else:
                    B_sqe[i] += matrix[at1][j]*list_of_q0[j] - matrix[at2][j]*list_of_q0[j]

            for j in range(i+1, molecule.num_of_bonds):
                at2_1, at2_2, _ = bonds[j]
                if at1 == at2_1:
                    A_sqe[i, j] = A_sqe[j, i] = -1*(matrix[at1][at2] + matrix[at2_1][at2_2] - list_of_hardness[at1] - matrix[at2][at2_2])

                elif at1 == at2_2:
                    A_sqe[i, j] = A_sqe[j, i] = (matrix[at1][at2] + matrix[at2_1][at2_2] - list_of_hardness[at1] - matrix[at2][at2_1])

                elif at2 == at2_1:
                    A_sqe[i, j] = A_sqe[j, i] = (matrix[at1][at2] + matrix[at2_1][at2_2] - list_of_hardness[at2] - matrix[at1][at2_2])

                elif at2 == at2_2:
                    A_sqe[i, j] = A_sqe[j, i] = -1*(matrix[at1][at2] + matrix[at2_1][at2_2] - list_of_hardness[at2] - matrix[at1][at2_1])

                else:
                   A_sqe[i, j] = A_sqe[j, i] = matrix[at1][at2_1] + matrix[at2][at2_2] - matrix[at2][at2_1] - matrix[at1][at2_2]
        # print(B_sqe)
        # exit()
        results[index: new_index] = np.dot(np.linalg.solve(A_sqe, -B_sqe), T) + list_of_q0
        index = new_index
    return results


class SQEqpopt(ChargeMethod):

    @property
    def params_pattern(self):
        return {"atom": {"data": {},
                         "names": ["chi", "eta", "width"]},
                "bond": {"data": {},
                         "name": "hardness"},
                "metadata": {"method": "SQEqpopt"}}

    @property
    def params_bounds(self):
        params_bounds = [[-0, 5] for _ in range(len(self.params_vals))]
        return np.array(params_bounds)

    def calculate(self, set_of_molecules):

        return sqeqpopt_calculate(set_of_molecules, self.params_vals, len(self.ats_types))




