from sys import exit
from termcolor import colored
from numba import jit
from numpy import float64, empty, array, zeros, sqrt, sum, random
from numpy.linalg import solve
from math import erf
from json import load


def convert_atom(atom):
    if isinstance(atom, list):
        if atom[1] == "hbo":
            return "{}~{}".format(atom[0], atom[2])
        elif atom[1] == "plain":
            return atom[0]
        elif atom[1] == "hbob":
            return "/".join([atom[0], atom[2]])
    elif isinstance(atom, str):
        if "/" in atom:
            s_atom = atom.split("/")
            return [s_atom[0], "hbob", s_atom[1]]
        s_atom = atom.split("~")
        if len(s_atom) == 2:
            return [s_atom[0], "hbo", s_atom[1]]
        elif len(s_atom) == 1:
            return [atom, "plain", "*"]


def convert_bond(bond):
    if isinstance(bond, list):
        bond_atoms = "-".join(*sorted([bond[:2]]))
        if bond[2] in ["hbo", "hbob"]:
            return "bond-{}-{}".format(bond_atoms, bond[3])
        elif bond[2] == "plain":
            return "bond-{}".format(bond_atoms)
    elif isinstance(bond, str):
        s_bond = bond.split("-")
        if "/" in bond:
            return [s_bond[0], s_bond[1], "hbob", bond[-1]]
        elif "~" in bond:
            return [s_bond[0], s_bond[1], "hbo", bond[-1]]
        else:
            return [s_bond[0], s_bond[1], "plain", "*"]


class Methods:

    def __repr__(self):
        return self.__class__.__name__

    def load_parameters(self, parameters_file, set_of_molecules, mode, atomic_types_pattern):
        if not parameters_file:
            parameters_file = "modules/parameters/{}.json".format(str(self))
        print("Loading of parameters from {}...".format(parameters_file))
        try:
            self.parameters_json = load(open(parameters_file))
        except FileNotFoundError:
            exit(colored("Error! There is no file {}.".format(parameters_file)), "red")
        self.atomic_types_pattern = atomic_types_pattern
        method_in_parameters_file = self.parameters_json["metadata"]["method"]
        if self.__class__.__name__ != method_in_parameters_file:
            exit(colored("These parameters are for method {} but you want to calculate charges by method {}!\n"
                         .format(method_in_parameters_file, self.__class__.__name__), "red"))
        self.atomic_parameters_types = self.parameters_json["atom"]["names"]
        set_of_molecules_atoms = set([atom for molecule in set_of_molecules.molecules for atom in molecule.atoms_representation])
        json_parameters_atoms = [parameters["key"] for parameters in self.parameters_json["atom"]["data"]]
        if "bond" in self.parameters_json:
            set_of_molecules_bonds = set([bond for molecule in set_of_molecules.molecules for bond in molecule.bonds_representation])
            json_parameters_bonds = [parameters["key"] for parameters in self.parameters_json["bond"]["data"]]
        if mode == "calculation":
            missing_atoms = []
            for atom in set_of_molecules_atoms:
                converted_atom = convert_atom(atom)
                if converted_atom not in json_parameters_atoms:
                    if "fixed" in self.parameters_json:
                        if atom not in self.parameters_json["fixed"].keys():
                            missing_atoms.append(atom)
                    else:
                        missing_atoms.append(atom)
            if missing_atoms:
                print(colored("Atomic type(s) {} is not defined in parameters.".format(", ".join(missing_atoms)), "red"))
            if "bond" in self.parameters_json:
                missing_bonds = []
                for bond in set_of_molecules_bonds:
                    converted_bond = convert_bond(bond)
                    if converted_bond not in json_parameters_bonds:
                        missing_bonds.append(bond)
                if missing_bonds:
                    exit(colored("Bond type(s) {} is not defined in parameters.".format(", ".join(missing_bonds)), "red"))
            if missing_atoms:
                exit()
        elif mode == "parameterization":
            for atom in set_of_molecules_atoms:
                converted_atom = convert_atom(atom)
                if converted_atom not in json_parameters_atoms:
                    if converted_atom[1] == "hbob":
                        for x in self.parameters_json["atom"]["data"]:
                            if "~".join([x["key"][0], x["key"][2]]) == converted_atom[0]:
                                vall = list(x["value"])
                                break
                        self.parameters_json["atom"]["data"].append({"key": converted_atom, "value": vall})
                    else:
                        self.parameters_json["atom"]["data"].append({"key": converted_atom, "value": [random.random() for _ in range(len(self.atomic_parameters_types))]})
                    print(colored("    Atom {} was added to parameters.".format(atom), "yellow"))
                else:
                    json_parameters_atoms.remove(converted_atom)
            for atom in json_parameters_atoms:
                for parameter in self.parameters_json["atom"]["data"]:
                    if parameter["key"] == atom:
                        self.parameters_json["atom"]["data"].remove(parameter)
                print(colored("    Atom {} was deleted from parameters.".format(convert_atom(atom)), "yellow"))
            if "bond" in self.parameters_json:
                for bond in set_of_molecules_bonds:
                    converted_bond = convert_bond(bond)
                    if converted_bond not in json_parameters_bonds:
                        if converted_bond[2] == "hbob":
                            for x in self.parameters_json["bond"]["data"]:
                                if x["key"][0] == converted_bond[0].split("/")[0] and x["key"][1] == converted_bond[1].split("/")[0] and x["key"][-1] == converted_bond[-1]:
                                    vall = list(x["value"])
                                    break
                            self.parameters_json["bond"]["data"].append({"key": converted_bond, "value": vall})
                        else:
                            self.parameters_json["bond"]["data"].append({"key": converted_bond, "value": [random.random()]})
                        print(colored("     Bond {} was added to parameters.".format(bond), "yellow"))
                    else:
                        json_parameters_bonds.remove(converted_bond)
                for bond in json_parameters_bonds:
                    for parameter in self.parameters_json["bond"]["data"]:
                        if parameter["key"] == bond:
                            self.parameters_json["bond"]["data"].remove(parameter)
                    print(colored("    Bond {} was deleted from parameters.".format(convert_bond(bond)[5:]), "yellow"))
        self.parameters = {}
        if "common" in self.parameters_json:
            for name, value in zip(self.parameters_json["common"]["names"], self.parameters_json["common"]["values"]):
                self.parameters[name] = value
        atomic_types = []
        for parameter in self.parameters_json["atom"]["data"]:
            atomic_type = convert_atom(parameter["key"])
            atomic_types.append(atomic_type)
            for parameter_name, value in zip(self.atomic_parameters_types, parameter["value"]):
                self.parameters["{}_{}".format(atomic_type, parameter_name)] = value
        if "fixed" in self.parameters_json:
            atomic_types.extend(self.parameters_json["fixed"].keys())
        self.atomic_types = sorted(atomic_types)
        if "bond" in self.parameters_json:
            for parameter in self.parameters_json["bond"]["data"]:
                self.parameters[convert_bond(parameter["key"])] = parameter["value"][0]
        self.bond_types = sorted([key for key in self.parameters.keys() if key[:5] == "bond-"])
        writed_glob_par = -1
        num_of_atom_par = len(self.atomic_parameters_types) * len(self.atomic_types)
        parameters_values = [0 for _ in range(len(self.parameters))]
        if "fixed" in self.parameters_json:
            parameters_values.extend([0 for x in range(2*len(self.parameters_json["fixed"]))])
        self.key_index = {}
        for key, value in self.parameters.items():
            if key[0].isupper():
                atomic_type, value_symbol = key.split("_")
                index = self.atomic_types.index(atomic_type) * len(self.atomic_parameters_types) + self.atomic_parameters_types.index(value_symbol)
                parameters_values[index] = value
                self.key_index[key] = index
            elif key[:5] == "bond-":
                index = num_of_atom_par + self.bond_types.index(key)
                parameters_values[index] = value
                self.key_index[key] = index
            else:
                parameters_values[writed_glob_par] = value
                self.key_index[key] = writed_glob_par
                writed_glob_par -= 1
        self.parameters_values = array(parameters_values, dtype=float64)
        self.bounds = [(min(self.parameters_values), max(self.parameters_values))] * len(self.parameters_values)
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
            for index, parameter in enumerate(self.atomic_parameters_types):
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
