from json import load
from math import erf
from sys import exit

from numba import jit
from numpy import float64, empty, array, zeros, sqrt, random, dot, exp
from numpy.linalg import solve
from termcolor import colored


class Method:
    def __repr__(self):
        return self.__class__.__name__

    def load_parameters(self, parameters_file, set_of_molecules, mode, atomic_types_pattern):
        if not parameters_file:
            parameters_file = "modules/parameters/{}.json".format(str(self))

        print(f"Loading of parameters from {parameters_file}...")
        self.parameters = load(open(parameters_file))

        method_in_parameters_file = self.parameters["metadata"]["method"]
        if self.__class__.__name__ != method_in_parameters_file:
            exit(colored(f"Error! These parameters are for method {method_in_parameters_file}, but you selected by argument by method {self.__class__.__name__}!\n", "red"))


        # missing atoms and missing bond are atomic types and bond types which are in set of molecules but not in parameters
        print(self.parameters["atom"]["data"])
        missing_atoms = set(set_of_molecules.atomic_types) - set(self.parameters["atom"]["data"].keys())
        if "bond" in self.parameters:
            missing_bonds = set(set_of_molecules.bonds_types) - set(self.parameters["bond"]["data"].keys())

        if mode == "calculation":
            exit_status = False
            if missing_atoms:
                print(colored(f"Error! Atomic type(s) {', '.join(missing_atoms)} is not defined in parameters.", "red"))
                exit_status = True
            if "bond" in self.parameters:
                if missing_bonds:
                    print(colored(f"Error! Bond type(s) {', '.join(missing_bonds)} is not defined in parameters.", "red"))
                    exit_status = True
            if exit_status:
                exit()

        elif mode == "parameterization":
            for atom in missing_atoms:
                for key, values in self.parameters["atom"]["data"].items():
                    if key.split("~")[0] == atom.split("~")[0]:
                        self.parameters["atom"]["data"][atom] = values
                        print(colored(f"    Atom type {atom} was added to parameters. Parameters derived from {key}", "yellow"))
                        break
                else:
                    self.parameters["atom"]["data"][atom] = [random.random() for _ in range(len(self.parameters["atom"]["names"]))]
                    print(colored(f"    Atom type {atom} was added to parameters. Parameters are random numbers.", "yellow"))
            # unused atoms are atomic types which are in parameters but not in set of molecules
            unused_atoms = set(self.parameters["atom"]["data"].keys()) - set(set_of_molecules.atomic_types)
            for atom in unused_atoms:
                del self.parameters["atom"]["data"][atom]
            if unused_atoms:
                print(colored(f"    Atom(s) {', '.join(unused_atoms)} was deleted from parameters.", "yellow"))

            if "bond" in self.parameters:
                for bond in missing_bonds:
                    for key, value in self.parameters["bond"]["data"].items():
                        patom1, patom2, ptype = key.split("-")
                        natom1, natom2, ntype = bond.split("-")
                        if patom1.split("/")[0] == natom1.split("/")[0] and patom2.split("/")[0] == natom2.split("/")[0] and ptype.split("/")[0] == ntype.split("/")[0]:
                            self.parameters["bond"]["data"][bond] = value
                            print(colored(f"    Bond type {bond} was added to parameters. Parameter derived from {key}", "yellow"))
                            break
                    else:
                        self.parameters["bond"]["data"][bond] = random.random()
                        print(colored(f"    Bond type {bond} was added to parameters. Parameters are random numbers.", "yellow"))
                # unused bonds are bond types which are in parameters but not in set of molecules
                unused_bonds = set(self.parameters["bond"]["data"].keys()) - set(set_of_molecules.bonds_types)
                for bond in unused_bonds:
                    del self.parameters["bond"]["data"][bond]
                if unused_bonds:
                    print(colored(f"    Bond(s) {', '.join(unused_bonds)} was deleted from parameters.", "yellow"))

        self.atomic_types = sorted(self.parameters["atom"]["data"].keys())
        try:
            self.bond_types = sorted(self.parameters["bond"]["data"].keys())
        except: # dodelat!!!!
            pass
        parameters_values = []
        for _, values in sorted(self.parameters["atom"]["data"].items()):
            parameters_values.extend(values)
        if "bond" in self.parameters:
            for _, value in sorted(self.parameters["bond"]["data"].items()):
                parameters_values.append(value)
        if "common" in self.parameters:
            parameters_values.extend(self.parameters["common"].values())
        self.parameters_values = array(parameters_values, dtype=float64)
        self.bounds = (min(self.parameters_values), max(self.parameters_values))
        print(colored("ok\n", "green"))


    def new_parameters(self, new_parameters):
        self.parameters_values = new_parameters
        parameters_per_atom = len(self.parameters["atom"]["names"])
        index = 0
        for key in self.atomic_types:
            self.parameters["atom"]["data"][key] = list(self.parameters_values[index: index + parameters_per_atom])
            index = index + parameters_per_atom

        if "bond" in self.parameters:
            for key in self.bond_types:
                self.parameters["bond"]["data"][key] = self.parameters_values[index]
                index += 1

        if "common" in self.parameters:
            for key in sorted(self.parameters["common"]):
                self.parameters["common"][key] = self.parameters_values[index]
                index += 1


@jit(nopython=True, cache=True)
def eem_calculate(set_of_molecules, parameters):
    results = empty(set_of_molecules.num_of_atoms, dtype=float64)
    kappa = parameters[-1]
    index = 0
    for molecule in set_of_molecules.molecules:
        num_of_atoms = molecule.num_of_atoms
        new_index = index + num_of_atoms
        matrix = empty((num_of_atoms + 1, num_of_atoms + 1), dtype=float64)
        vector = empty(num_of_atoms + 1, dtype=float64)
        matrix[:num_of_atoms, :num_of_atoms] = kappa / molecule.distance_matrix
        matrix[num_of_atoms, :] = 1.0
        matrix[:, num_of_atoms] = 1.0
        matrix[num_of_atoms, num_of_atoms] = 0.0
        for x, parameter_index in enumerate(molecule.atoms_id):
            matrix[x, x] = parameters[parameter_index + 1]
            vector[x] = -parameters[parameter_index]
        vector[-1] = molecule.total_charge
        results[index: new_index] = solve(matrix, vector)[:-1]
        index = new_index

    return results


class EEM(Method):
    def calculate(self, set_of_molecules):

        from time import time


        for x in range(10):
            start = time()


            self.results = eem_calculate(set_of_molecules, self.parameters_values)
            print(time() - start)



@jit(nopython=True, cache=True)
def qeq_calculate(set_of_molecules, parameters, num_of_atomic_types):
    rad_values = empty((num_of_atomic_types * 3, num_of_atomic_types * 3), dtype=float64)
    for x in range(num_of_atomic_types):
        for y in range(num_of_atomic_types):
            rad_values[x * 3][y * 3] = sqrt(parameters[x*3+2] * parameters[y*3+2] / (parameters[x*3+2] + parameters[y*3+2]))

    results = empty(set_of_molecules.num_of_atoms, dtype=float64)
    index = 0
    for molecule in set_of_molecules.molecules:
        num_of_atoms = molecule.num_of_atoms
        new_index = index + num_of_atoms
        matrix = empty((num_of_atoms + 1, num_of_atoms + 1), dtype=float64)
        vector = empty(num_of_atoms + 1, dtype=float64)
        matrix[num_of_atoms, :] = 1.0
        matrix[:, num_of_atoms] = 1.0
        matrix[num_of_atoms, num_of_atoms] = 0.0
        for i, parameter_index_i in enumerate(molecule.atoms_id):
            matrix[i, i] = parameters[parameter_index_i + 1]
            vector[i] = -parameters[parameter_index_i]
            for j, (parameter_index_j, distance) in enumerate(zip(molecule.atoms_id[i + 1:], molecule.distance_matrix[i, i + 1:]), i + 1):
                matrix[i, j] = matrix[j, i] = erf(rad_values[parameter_index_i, parameter_index_j] * distance) / distance
        vector[-1] = molecule.total_charge
        results[index: new_index] = solve(matrix, vector)[:-1]
        index = new_index
    return results


class QEq(Method):
    def calculate(self, set_of_molecules):
        self.results = qeq_calculate(set_of_molecules, self.parameters_values, len(self.atomic_types))


@jit(nopython=True, cache=True)
def eqeq_calculate(set_of_molecules, parameters, num_of_atomic_types):
    k_parameter = parameters[-2]
    lambda_parameter = parameters[-1]
    J_values = empty(num_of_atomic_types*2, dtype=float64)
    X_values = empty(num_of_atomic_types*2, dtype=float64)
    for x in range(num_of_atomic_types):
        J_values[x*2] = parameters[x*2] - parameters[x*2 + 1]
        X_values[x*2] = (parameters[x*2] + parameters[x*2 + 1]) / 2
    a_values = empty((num_of_atomic_types * 2, num_of_atomic_types * 2), dtype=float64)
    for x in range(num_of_atomic_types):
        for y in range(num_of_atomic_types):
            a_values[x * 2][y * 2] = sqrt(J_values[x*2] * J_values[y*2]) / k_parameter
    results = empty(set_of_molecules.num_of_atoms, dtype=float64)
    index = 0
    for molecule in set_of_molecules.molecules:
        num_of_atoms = molecule.num_of_atoms
        new_index = index + num_of_atoms
        matrix = empty((num_of_atoms + 1, num_of_atoms + 1), dtype=float64)
        vector = empty(num_of_atoms + 1, dtype=float64)
        matrix[num_of_atoms, :] = 1.0
        matrix[:, num_of_atoms] = 1.0
        matrix[num_of_atoms, num_of_atoms] = 0.0
        for i, parameter_index_i in enumerate(molecule.atoms_id):
            matrix[i, i] = J_values[parameter_index_i]
            vector[i] = -X_values[parameter_index_i]
            for j, (parameter_index_j, distance) in enumerate(zip(molecule.atoms_id[i + 1:], molecule.distance_matrix[i, i + 1:]), i + 1):
                a = a_values[parameter_index_i, parameter_index_j]
                overlap = exp(-a * a * distance**2) * (2 * a - a * a * distance - 1 / distance)
                matrix[i, j] = matrix[j, i] = lambda_parameter * k_parameter / 2 * (1 / distance + overlap)
        vector[-1] = molecule.total_charge
        results[index: new_index] = solve(matrix, vector)[:-1]
        index = new_index
    return results

class EQEq(Method):
    def calculate(self, set_of_molecules):
        self.results = eqeq_calculate(set_of_molecules, self.parameters_values, len(self.atomic_types))


@jit(nopython=True, cache=True)
def sqe_calculate(set_of_molecules, parameters, num_of_atomic_types):
    multiplied_widths = empty((num_of_atomic_types * 4, num_of_atomic_types * 4), dtype=float64)
    for x in range(num_of_atomic_types):
        for y in range(num_of_atomic_types):
            multiplied_widths[x * 4][y * 4] = sqrt(2 * parameters[x * 4 + 2] ** 2 + 2 * parameters[y * 4 + 2] ** 2)
    results = empty(set_of_molecules.num_of_atoms, dtype=float64)
    index = 0
    for molecule in set_of_molecules.molecules:
        num_of_atoms = molecule.num_of_atoms
        new_index = index + num_of_atoms
        T = zeros((len(molecule.bonds), num_of_atoms))
        bonds = molecule.bonds
        for i in range(len(bonds)):
            atom1, atom2, _ = bonds[i]
            T[i, atom1] += 1
            T[i, atom2] -= 1
        matrix = zeros((num_of_atoms, num_of_atoms))
        vector = zeros(num_of_atoms)
        list_of_q0 = empty(num_of_atoms, dtype=float64)
        list_of_hardness = empty(num_of_atoms, dtype=float64)
        for i, parameter_index_i in enumerate(molecule.atoms_id):
            matrix[i, i] = parameters[parameter_index_i + 1]
            list_of_hardness[i] = parameters[parameter_index_i + 1]
            vector[i] = -parameters[parameter_index_i]
            list_of_q0[i] = parameters[parameter_index_i + 3]
            for j, (parameter_index_j, distance) in enumerate(zip(molecule.atoms_id[i + 1:], molecule.distance_matrix[i, i + 1:]), i + 1):
                d0 = multiplied_widths[parameter_index_i, parameter_index_j]
                matrix[i, j] = matrix[j, i] = erf(distance / d0) / distance
        vector -= dot(matrix, list_of_q0)
        vector += list_of_hardness*list_of_q0
        A_sqe = dot(T, dot(matrix, T.T))
        B_sqe = dot(T, vector)
        for i, bond_parameters_index in enumerate(molecule.bonds_id):
            A_sqe[i, i] += parameters[bond_parameters_index]
        results[index: new_index] = dot(solve(A_sqe, B_sqe), T) + list_of_q0
        index = new_index
    return results


class SQE(Method):
    def calculate(self, set_of_molecules):
        self.results = sqe_calculate(set_of_molecules, self.parameters_values, len(self.atomic_types))

