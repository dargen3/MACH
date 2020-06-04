from json import load
from math import erf
from sys import exit

import numpy as np
from numba import jit
from termcolor import colored


class ChargeMethod:
    def __repr__(self):
        return self.__class__.__name__

    def load_params(self,
                    params_file: str) -> str:

        print(f"Loading of parameters from {params_file}...")
        self.params = load(open(params_file))
        self.at_types_pattern = self.params["metadata"]["atomic_types_pattern"]
        method_in_params_file = self.params["metadata"]["method"]
        if self.__class__.__name__ != method_in_params_file:
            exit(colored(f"ERROR! These parameters are for method {method_in_params_file},"
                         f"but you selected by argument --chg_method {self.__class__.__name__}!\n", "red"))
        print(colored("ok\n", "green"))
        return self.at_types_pattern

    def prepare_params_for_calc(self,
                                set_of_molecules: "SetOfMolecules"):
        missing_at = set(set_of_molecules.at_types) - set(self.params["atom"]["data"].keys())
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
        missing_at = set(set_of_molecules.at_types) - set(self.params["atom"]["data"].keys())
        for at in missing_at:
            for key, vals in self.params["atom"]["data"].items():
                if key.split("/")[0] == at.split("/")[0]:
                    self.params["atom"]["data"][at] = vals
                    print(colored(f"    Atom type {at} was added to parameters. Parameters derived from {key}", "yellow"))
                    break
            else:
                self.params["atom"]["data"][at] = [np.random.random() for _ in range(len(self.params["atom"]["names"]))]
                print(colored(f"    Atom type {at} was added to parameters. Parameters are random numbers.", "yellow"))
        # unused atoms are atomic types which are in parameters but not in set of molecules
        unused_at = set(self.params["atom"]["data"].keys()) - set(set_of_molecules.at_types)
        for at in unused_at:
            del self.params["atom"]["data"][at]
        if unused_at:
            print(colored(f"    Atom(s) {', '.join(unused_at)} was deleted from parameters.", "yellow"))

        if "bond" in self.params:
            missing_bonds = set(set_of_molecules.bonds_types) - set(self.params["bond"]["data"].keys())
            for bond in missing_bonds:
                nat1, nat2, ntype = bond.split("-")
                for key, val in self.params["bond"]["data"].items():
                    pat1, pat2, ptype = key.split("-")
                    if pat1.split("/")[0] == nat1.split("/")[0] and pat2.split("/")[0] == nat2.split("/")[0] and ptype.split("/")[0] == ntype.split("/")[0]:
                        self.params["bond"]["data"][bond] = val
                        print(colored(f"    Bond type {bond} was added to parameters. Parameter derived from {key}", "yellow"))
                        break
                else:
                    self.params["bond"]["data"][bond] = np.random.random()
                    print(colored(f"    Bond type {bond} was added to parameters. Parameter is random numbers.", "yellow"))
            # unused bonds are bond types which are in parameters but not in set of molecules
            unused_bonds = set(self.params["bond"]["data"].keys()) - set(set_of_molecules.bonds_types)
            for bond in unused_bonds:
                del self.params["bond"]["data"][bond]
            if unused_bonds:
                print(colored(f"    Bond(s) {', '.join(unused_bonds)} was deleted from parameters.", "yellow"))
        self._dict_to_array()

    def _dict_to_array(self):
        self.at_types = sorted(self.params["atom"]["data"].keys())
        if "bond" in self.params:
            self.bond_types = sorted(self.params["bond"]["data"].keys())
        params_vals = []
        for _, vals in sorted(self.params["atom"]["data"].items()):
            params_vals.extend(vals)
        if "bond" in self.params:
            for _, val in sorted(self.params["bond"]["data"].items()):
                params_vals.append(val)
        if "common" in self.params:
            params_vals.extend(self.params["common"].vals())
        self.params_vals = np.array(params_vals, dtype=np.float64)
        self.bounds = (min(self.params_vals), max(self.params_vals))

    def new_params(self,
                   new_params: np.array):
        self.params_vals = new_params
        params_per_at = len(self.params["atom"]["names"])
        index = 0
        for at in self.at_types:
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


@jit(nopython=True, cache=True)
def eem_calculate(set_of_molecules, params):
    results = np.empty(set_of_molecules.num_of_at, dtype=np.float64)
    kappa = params[-1]
    index = 0
    for molecule in set_of_molecules.molecules:
        num_of_at = molecule.num_of_at
        new_index = index + num_of_at
        matrix = np.empty((num_of_at + 1, num_of_at + 1), dtype=np.float64)
        vector = np.empty(num_of_at + 1, dtype=np.float64)
        matrix[:num_of_at, :num_of_at] = kappa / molecule.distance_matrix
        matrix[num_of_at, :] = 1.0
        matrix[:, num_of_at] = 1.0
        matrix[num_of_at, num_of_at] = 0.0
        for x, parameter_index in enumerate(molecule.at_id):
            matrix[x, x] = params[parameter_index + 1]
            vector[x] = -params[parameter_index]
        vector[-1] = molecule.total_charge
        results[index: new_index] = np.linalg.solve(matrix, vector)[:-1]
        index = new_index

    return results


class EEM(ChargeMethod):
    def calculate(self, set_of_molecules):
        return eem_calculate(set_of_molecules, self.params_vals)


@jit(nopython=True, cache=True)
def qeq_calculate(set_of_molecules, params, num_of_at_types):
    rad_vals = np.empty((num_of_at_types * 3, num_of_at_types * 3), dtype=np.float64)
    for x in range(num_of_at_types):
        for y in range(num_of_at_types):
            rad_vals[x * 3][y * 3] = np.sqrt(params[x*3+2] * params[y*3+2] / (params[x*3+2] + params[y*3+2]))

    results = np.empty(set_of_molecules.num_of_at, dtype=np.float64)
    index = 0
    for molecule in set_of_molecules.molecules:
        num_of_at = molecule.num_of_at
        new_index = index + num_of_at
        matrix = np.empty((num_of_at + 1, num_of_at + 1), dtype=np.float64)
        vector = np.empty(num_of_at + 1, dtype=np.float64)
        matrix[num_of_at, :] = 1.0
        matrix[:, num_of_at] = 1.0
        matrix[num_of_at, num_of_at] = 0.0
        for i, parameter_index_i in enumerate(molecule.at_id):
            matrix[i, i] = params[parameter_index_i + 1]
            vector[i] = -params[parameter_index_i]
            for j, (parameter_index_j, distance) in enumerate(zip(molecule.at_id[i + 1:], molecule.distance_matrix[i, i + 1:]), i + 1):
                matrix[i, j] = matrix[j, i] = erf(rad_vals[parameter_index_i, parameter_index_j] * distance) / distance
        vector[-1] = molecule.total_charge
        results[index: new_index] = np.linalg.solve(matrix, vector)[:-1]
        index = new_index
    return results


class QEq(ChargeMethod):
    def calculate(self, set_of_molecules):
        return qeq_calculate(set_of_molecules, self.params_vals, len(self.at_types))


@jit(nopython=True, cache=True)
def eqeq_calculate(set_of_molecules, params, num_of_at_types):
    k_parameter = params[-2]
    lambda_parameter = params[-1]
    J_vals = np.empty(num_of_at_types*2, dtype=np.float64)
    X_vals = np.empty(num_of_at_types*2, dtype=np.float64)
    for x in range(num_of_at_types):
        J_vals[x*2] = params[x*2] - params[x*2 + 1]
        X_vals[x*2] = (params[x*2] + params[x*2 + 1]) / 2
    a_vals = np.empty((num_of_at_types * 2, num_of_at_types * 2), dtype=np.float64)
    for x in range(num_of_at_types):
        for y in range(num_of_at_types):
            a_vals[x * 2][y * 2] = np.sqrt(J_vals[x*2] * J_vals[y*2]) / k_parameter
    results = np.empty(set_of_molecules.num_of_at, dtype=np.float64)
    index = 0
    for molecule in set_of_molecules.molecules:
        num_of_at = molecule.num_of_at
        new_index = index + num_of_at
        matrix = np.empty((num_of_at + 1, num_of_at + 1), dtype=np.float64)
        vector = np.empty(num_of_at + 1, dtype=np.float64)
        matrix[num_of_at, :] = 1.0
        matrix[:, num_of_at] = 1.0
        matrix[num_of_at, num_of_at] = 0.0
        for i, parameter_index_i in enumerate(molecule.at_id):
            matrix[i, i] = J_vals[parameter_index_i]
            vector[i] = -X_vals[parameter_index_i]
            for j, (parameter_index_j, distance) in enumerate(zip(molecule.at_id[i + 1:], molecule.distance_matrix[i, i + 1:]), i + 1):
                a = a_vals[parameter_index_i, parameter_index_j]
                overlap = np.exp(-a * a * distance**2) * (2 * a - a * a * distance - 1 / distance)
                matrix[i, j] = matrix[j, i] = lambda_parameter * k_parameter / 2 * (1 / distance + overlap)
        vector[-1] = molecule.total_charge
        results[index: new_index] = np.linalg.solve(matrix, vector)[:-1]
        index = new_index
    return results


class EQEq(ChargeMethod):
    def calculate(self, set_of_molecules):
        return eqeq_calculate(set_of_molecules, self.params_vals, len(self.at_types))


@jit(nopython=True, cache=True)
def sqe_calculate(set_of_molecules, params, num_of_at_types):
    multiplied_widths = np.empty((num_of_at_types * 4, num_of_at_types * 4), dtype=np.float64)
    for x in range(num_of_at_types):
        for y in range(num_of_at_types):
            multiplied_widths[x * 4][y * 4] = np.sqrt(2 * params[x * 4 + 2] ** 2 + 2 * params[y * 4 + 2] ** 2)
    results = np.empty(set_of_molecules.num_of_at, dtype=np.float64)
    index = 0
    for molecule in set_of_molecules.molecules:
        num_of_at = molecule.num_of_at
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
        for i, parameter_index_i in enumerate(molecule.at_id):
            matrix[i, i] = params[parameter_index_i + 1]
            list_of_hardness[i] = params[parameter_index_i + 1]
            vector[i] = -params[parameter_index_i]
            list_of_q0[i] = params[parameter_index_i + 3]
            for j, (parameter_index_j, distance) in enumerate(zip(molecule.at_id[i + 1:], molecule.distance_matrix[i, i + 1:]), i + 1):
                d0 = multiplied_widths[parameter_index_i, parameter_index_j]
                matrix[i, j] = matrix[j, i] = erf(distance / d0) / distance
        vector -= np.dot(matrix, list_of_q0)
        vector += list_of_hardness*list_of_q0
        A_sqe = np.dot(T, np.dot(matrix, T.T))
        B_sqe = np.dot(T, vector)
        for i, bond_params_index in enumerate(molecule.bonds_id):
            A_sqe[i, i] += params[bond_params_index]
        results[index: new_index] = np.dot(np.linalg.solve(A_sqe, B_sqe), T) + list_of_q0
        index = new_index
    return results


class SQE(ChargeMethod):
    def calculate(self, set_of_molecules):
        return sqe_calculate(set_of_molecules, self.params_vals, len(self.at_types))
