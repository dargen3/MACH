from sys import exit
from termcolor import colored
from numba import jit
from numpy import float64, empty, array, zeros, sqrt, sum, random
from numpy.linalg import solve
from math import erf
from json import load
from .convert_parameters import convert_parameters_schindler


class Methods:
    def __repr__(self):
        return self.__class__.__name__

    def load_parameters(self, parameters_file, set_of_molecules, mode, atomic_types_pattern):
        if not parameters_file:
            parameters_file = "modules/parameters/{}.json".format(str(self))

        print(f"Loading of parameters from {parameters_file}...")
        try:
            self.parameters = load(open(parameters_file))
        except FileNotFoundError:
            exit(colored(f"Error! There is no file {parameters_file}.", "red"))

        method_in_parameters_file = self.parameters["metadata"]["method"]
        if self.__class__.__name__ != method_in_parameters_file:
            exit(colored(f"Error! These parameters are for method {method_in_parameters_file}, but you selected by argument by method {self.__class__.__name__}!\n", "red"))

        if isinstance(self.parameters["atom"]["data"], list):
            self.parameters = convert_parameters_schindler(self.parameters)

        if self.parameters["metadata"]["atomic_types_pattern"] != atomic_types_pattern:
            exit(colored(f"Error! These parameters are for atomic types pattern {self.parameters['metadata']['atomic_types_pattern']}, but you selected by argument atomic types pattern {atomic_types_pattern}!\n", "red"))

        # missing atoms and missing bond are atomic types and bond types which are in set of molecules but not in parameters
        missing_atoms = set(set_of_molecules.atomic_types) - set(self.parameters["atom"]["data"].keys())
        if "bond" in self.parameters:
            missing_bonds = set(set_of_molecules.bond_types) - set(self.parameters["bond"]["data"].keys())

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
                self.parameters["atom"]["data"][atom] = [random.random() for _ in range(len(self.parameters["atom"]["names"]))]
            if missing_atoms:
                print(colored(f"    Atom(s) {', '.join(missing_atoms)} was added to parameters.", "yellow"))
            # unused atoms are atomic types which are in parameters but not in set of molecules
            unused_atoms = set(self.parameters["atom"]["data"].keys()) - set(set_of_molecules.atomic_types)
            for atom in unused_atoms:
                del self.parameters["atom"]["data"][atom]
            if unused_atoms:
                print(colored(f"    Atom(s) {', '.join(unused_atoms)} was deleted from parameters.", "yellow"))

            if "bond" in self.parameters:
                for bond in missing_bonds:
                    self.parameters["bond"]["data"][bond] = random.random()
                if missing_bonds:
                    print(colored(f"    Bonds(s) {', '.join(missing_bonds)} was added to parameters.", "yellow"))
                # unused bonds are bond types which are in parameters but not in set of molecules
                unused_bonds = set(self.parameters["bond"]["data"].keys()) - set(set_of_molecules.bond_types)
                for bond in unused_bonds:
                    del self.parameters["bond"]["data"][bond]
                if unused_bonds:
                    print(colored(f"    Bond(s) {', '.join(unused_bonds)} was deleted from parameters.", "yellow"))

        parameters_values = []
        for parameter, values in sorted(self.parameters["atom"]["data"].items()):
            parameters_values.extend(values)
        if "bond" in self.parameters:
            for parameter, value in sorted(self.parameters["bond"]["data"].items()):
                parameters_values.append(value)
        if "common" in self.parameters:
            parameters_values.extend(self.parameters["common"].values())
        self.parameters_values = array(parameters_values, dtype=float64)
        self.bounds = (min(self.parameters_values), max(self.parameters_values))
        print(colored("ok\n", "green"))

    def new_parameters(self, new_parameters):
        self.parameters_values = new_parameters
        parameters = {}
        for key in self.parameters.keys():
            parameters[key] = self.parameters_values[self.key_index[key]]
        self.parameters = parameters
        if "common" in self.parameters_json:
            for index, global_parameter in enumerate(self.parameters_json["common"]["names"]):
                self.parameters_json["common"]["values"][index] = self.parameters[global_parameter]
        for atomic_type in self.parameters_json["atom"]["data"]:
            for index, parameter in enumerate(self.parameters["atom"]["names"]):
                atomic_type["value"][index] = self.parameters["{}_{}".format(convert_atom(atomic_type["key"]), parameter)]
        if "bond" in self.parameters_json:
            for bond in self.parameters_json["bond"]["data"]:
                bond["value"][0] = self.parameters[convert_bond(bond["key"])]


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
            matrix[x][x] = parameters[parameter_index + 1]
            vector[x] = -parameters[parameter_index]
        vector[-1] = 0  # formal charge
        results[index: new_index] = solve(matrix, vector)[:-1]
        index = new_index
    return results


class EEM(Methods):
    def calculate(self, set_of_molecules):
        self.results = eem_calculate(set_of_molecules, self.parameters_values)


@jit(nopython=True, cache=True)
def qeq_calculate(set_of_molecules, parameters):
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
        vector_rad = empty(num_of_atoms, dtype=float64)
        distance_matrix = molecule.distance_matrix
        for x, parameter_index in enumerate(molecule.atoms_id):
            vector_rad[x] = parameters[parameter_index + 2]
        for i, parameter_index in enumerate(molecule.atoms_id):
            matrix[i][i] = parameters[parameter_index + 1]
            vector[i] = -parameters[parameter_index]
            rad1 = vector_rad[i]
            for j, (rad2, distance) in enumerate(zip(vector_rad[i + 1:], distance_matrix[i, i+1:]),i + 1):
                matrix[i][j] = matrix[j][i] = erf(sqrt(rad1 * rad2 / (rad1 + rad2)) * distance) / distance
        vector[-1] = 0  # formal charge
        results[index: new_index] = solve(matrix, vector)[:-1]
        index = new_index
    return results


class QEq(Methods):
    def calculate(self, set_of_molecules):
        self.results = qeq_calculate(set_of_molecules, self.parameters_values)


@jit(nopython=True, cache=True, fastmath=True)
def acks2_calculate(set_of_molecules, parameters):
    multiplied_widths = empty((len(set_of_molecules.atomic_types)*4, len(set_of_molecules.atomic_types)*4), dtype=float64)
    for x in range(len(set_of_molecules.atomic_types)):
        for y in range(len(set_of_molecules.atomic_types)):
            multiplied_widths[x*4][y*4] = sqrt(2 * parameters[x*4 + 2] ** 2 + 2 * parameters[y*4 + 2] ** 2)

    results = empty(set_of_molecules.num_of_atoms, dtype=float64)
    index = 0
    for molecule in set_of_molecules.molecules:
        bonds = molecule.bonds
        num_of_atoms = molecule.num_of_atoms
        new_index = index + num_of_atoms
        distance_matrix = molecule.distance_matrix
        matrix = zeros((2 * num_of_atoms + 2, 2 * num_of_atoms + 2))
        vector = zeros(2 * num_of_atoms + 2)
        list_of_q0 = empty(num_of_atoms, dtype=float64)
        list_of_eta = empty(num_of_atoms, dtype=float64)
        for cc, parameter_index in enumerate(molecule.atoms_id):
            eta = parameters[parameter_index + 1]
            matrix[cc][cc] = eta
            list_of_eta[cc] = eta
            vector[cc] = -parameters[parameter_index]
            list_of_q0[cc] = parameters[parameter_index + 3]
            for j, parameter_index_2 in enumerate(molecule.atoms_id[cc + 1:], cc + 1):
                d = distance_matrix[cc][j]
                d0 = multiplied_widths[parameter_index][parameter_index_2]
                matrix[cc, j] = matrix[j, cc] = erf(d/d0)/d
        vector[:num_of_atoms] += list_of_eta * list_of_q0
        matrix[num_of_atoms, :num_of_atoms] = 1
        matrix[:num_of_atoms, num_of_atoms] = 1
        matrix[-1, num_of_atoms + 1:2 * num_of_atoms + 1] = 1
        matrix[num_of_atoms + 1:2 * num_of_atoms + 1, -1] = 1
        vector[num_of_atoms] = sum(list_of_q0)
        vector[num_of_atoms + 1:2 * num_of_atoms + 1] = list_of_q0
        for p in range(num_of_atoms):
            matrix[p, num_of_atoms + 1 + p] = 1.0
            matrix[num_of_atoms + 1 + p, p] = 1.0
        for b_index, bond_id in enumerate(molecule.bonds_id):
            bond = bonds[b_index]
            i = num_of_atoms + 1 + bond[0]
            j = num_of_atoms + 1 + bond[1]
            bsoft = 1 / parameters[bond_id]
            matrix[i, j] += bsoft
            matrix[j, i] += bsoft
            matrix[i, i] -= bsoft
            matrix[j, j] -= bsoft
        results[index: new_index] = solve(matrix, vector)[:num_of_atoms]
        index = new_index
    return results


class ACKS2(Methods):
    def calculate(self, set_of_molecules):
        self.results = acks2_calculate(set_of_molecules, self.parameters_values)
