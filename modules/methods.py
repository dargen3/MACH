from sys import exit
from termcolor import colored
from numba import jit
from numpy import float64, empty, array, ones, zeros, sqrt, cosh, concatenate, int64, sum, prod, dot, delete, insert, random
from numpy.linalg import solve, eigvalsh
from math import erf
from json import load
from collections import Counter


def convert_atom(atom):
    if isinstance(atom, list):
        if atom[1] == "hbo":
            return "{}~{}".format(atom[0], atom[2])
        elif atom[1] == "plain":
            return atom[0]
        #elif atom[1] == "aba":
        #    return "{}~{}.{}".format(atom[0], atom[2], atom[3])
        #elif atom[1] == "abaa":
        #    return "{}#{}".format(atom[0], atom[2])
    elif isinstance(atom, str):
        #if "." in atom:
        #    sp = atom.split(".")
        #    return [atom.split("~")[0], "aba", sp[0].split("~")[1], sp[1]]
        #elif "_" in atom:
        #    sp = atom.split("_")
        #    return [sp[0], "abaa", sp[1]]
        s_atom = atom.split("~")
        if len(s_atom) == 2:
            return [s_atom[0], "hbo", s_atom[1]]
        elif len(s_atom) == 1:
            return [atom, "plain", "*"]


def convert_bond(bond):
    if isinstance(bond, list):
        bond_atoms = "-".join(*sorted([bond[:2]]))
        if bond[2] == "hbo":
            return "bond-{}-{}".format(bond_atoms, bond[3])
        elif bond[2] == "plain":
            return "bond-{}".format(bond_atoms)
    elif isinstance(bond, str):
        s_bond = bond.split("-")
        if "~" in bond:
            return [s_bond[0], s_bond[1], "hbo", bond[-1]]
        else:
            return [s_bond[0], s_bond[1], "plain", "*"]



class Methods:
    def __init__(self):
        self.necessarily_data = {"EEM": ["distances"],
                                 "EEMfixed": ["distances"],
                                 "QEq": ["distances"],
                                 "SFKEEM": ["distances"],
                                 "GM": ["bonds_without_bond_type", "num_of_bonds_mul_two"],
                                 "MGC": ["MGC_matrix"],
                                 "SQE": ["distances", "num_of_bonds_mul_two", "bonds_without_bond_type"],
                                 "ACKS2": ["distances", "num_of_bonds_mul_two", "bonds_without_bond_type"],
                                 }[str(self)]

    def __repr__(self):
        return self.__class__.__name__

    def load_parameters(self, parameters_file, set_of_molecules, mode, atomic_types_pattern):
        if not parameters_file:
            parameters_file = "modules/parameters/{}.json".format(str(self))
        print("Loading of parameters from {}...".format(parameters_file))
        self.parameters_json = load(open(parameters_file))
        self.atomic_types_pattern = atomic_types_pattern
        method_in_parameters_file = self.parameters_json["metadata"]["method"]
        if self.__class__.__name__ != method_in_parameters_file:
            exit(colored("These parameters are for method {} but you want to calculate charges by method {}!\n"
                         .format(method_in_parameters_file, self.__class__.__name__), "red"))
        self.atomic_parameters_types = self.parameters_json["atom"]["names"]
        set_of_molecules_atoms = set([atom for molecule in set_of_molecules for atom in molecule.atoms_representation(self.atomic_types_pattern)])
        json_parameters_atoms = [parameters["key"] for parameters in self.parameters_json["atom"]["data"]]
        if "bond" in self.parameters_json:
            bond_pattern = "{}_{}".format(self.atomic_types_pattern, self.atomic_types_pattern)
            set_of_molecules_bonds = set([bond for molecule in set_of_molecules for bond in molecule.bonds_representation(bond_pattern)])
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
                    self.parameters_json["atom"]["data"].append({"key": converted_atom, "value": [random.random() for _ in range(len(self.atomic_parameters_types))]})
                    print(colored("Atom {} was added to parameters.".format(atom), "yellow"))
                else:
                    json_parameters_atoms.remove(converted_atom)
            for atom in json_parameters_atoms:
                for parameter in self.parameters_json["atom"]["data"]:
                    if parameter["key"] == atom:
                        self.parameters_json["atom"]["data"].remove(parameter)
                print(colored("Atom {} was deleted from parameters.".format(convert_atom(atom)), "yellow"))
            if "bond" in self.parameters_json:
                for bond in set_of_molecules_bonds:
                    converted_bond = convert_bond(bond)
                    if converted_bond not in json_parameters_bonds:
                        self.parameters_json["bond"]["data"].append({"key": converted_bond, "value": [random.random()]})
                        print(colored("Bond {} was added to parameters.".format(bond) ,"yellow"))
                    else:
                        json_parameters_bonds.remove(converted_bond)
                for bond in json_parameters_bonds:
                    for parameter in self.parameters_json["bond"]["data"]:
                        if parameter["key"] == bond:
                            self.parameters_json["bond"]["data"].remove(parameter)
                    print(colored("Bond {} was deleted from parameters.".format(convert_bond(bond)[5:]) ,"yellow"))

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


#####################################################################

from numpy.linalg import inv
from numpy import dot

@jit(nopython=True, cache=True)
def eem_calculate(distances, symbols, all_num_of_atoms, parameters_values):
    results = empty(symbols.size, dtype=float64)
    kappa = parameters_values[-1]
    formal_charge = 0
    index = 0
    counter_distance = 0
    counter_symbols = 0
    for num_of_atoms in all_num_of_atoms:
        new_index = index + num_of_atoms
        num_of_atoms_add_1 = num_of_atoms + 1
        matrix = empty((num_of_atoms_add_1, num_of_atoms_add_1), dtype=float64)
        vector = empty(num_of_atoms_add_1, dtype=float64)
        for x in range(num_of_atoms):
            matrix[num_of_atoms][x] = matrix[x][num_of_atoms] = 1.0
            symbol = symbols[counter_symbols]
            counter_symbols += 1
            matrix[x][x] = parameters_values[symbol + 1]
            vector[x] = -parameters_values[symbol]
            for y in range(x+1, num_of_atoms):
                matrix[x][y] = matrix[y][x] = kappa / distances[counter_distance]
                counter_distance += 1
        matrix[num_of_atoms, num_of_atoms] = 0.0
        vector[-1] = formal_charge
        results[index: new_index] = solve(matrix, vector)[:-1]
        index = new_index
    return results


class EEM(Methods):
    def calculate(self, set_of_molecules):
        self.results = eem_calculate(set_of_molecules.all_distances, set_of_molecules.multiplied_all_symbolic_numbers_atoms,
                                     set_of_molecules.all_num_of_atoms, self.parameters_values)


##########################################################################################

class EEMfixed(Methods):
    def calculate(self, set_of_molecules):
        try:
            fixed_atoms_symbols = [self.atomic_types.index(x) for x in self.parameters_json["fixed"].keys()]
        except:
            fixed_atoms_symbols = []
        distances, symbols, all_num_of_atoms, parameters_values = set_of_molecules.all_distances, set_of_molecules.multiplied_all_symbolic_numbers_atoms,                            set_of_molecules.all_num_of_atoms, self.parameters_values
        results = empty(symbols.size, dtype=float64)
        kappa = parameters_values[-1]
        index = 0
        counter_distance = 0
        counter_symbols = 0
        for num_of_atoms in all_num_of_atoms:
            new_index = index + num_of_atoms
            num_of_atoms_add_1 = num_of_atoms + 1
            matrix = empty((num_of_atoms_add_1, num_of_atoms_add_1), dtype=float64)
            vector = empty(num_of_atoms_add_1, dtype=float64)
            fixed_atoms_index = []
            for x in range(num_of_atoms):
                matrix[num_of_atoms][x] = matrix[x][num_of_atoms] = 1.0
                symbol = symbols[counter_symbols]
                if symbol/2 in fixed_atoms_symbols:
                    fixed_atoms_index.append([x, symbol])
                counter_symbols += 1
                matrix[x][x] = parameters_values[symbol + 1]
                vector[x] = -parameters_values[symbol]
                for y in range(x + 1, num_of_atoms):
                    matrix[x][y] = matrix[y][x] = kappa / distances[counter_distance]
                    counter_distance += 1
            matrix[num_of_atoms, num_of_atoms] = 0.0
            for fixed_atom in fixed_atoms_index[::]:
                vector -= matrix[:,fixed_atom[0]]
            vector = delete(vector, [fixed_atom[0] for fixed_atom in fixed_atoms_index], axis=0)
            matrix = delete(matrix, [fixed_atom[0] for fixed_atom in fixed_atoms_index], axis=0)
            matrix = delete(matrix, [fixed_atom[0] for fixed_atom in fixed_atoms_index], axis=1)
            vector[-1] = 0
            for fixed_atom in fixed_atoms_index:
                vector[-1] -= self.parameters_json["fixed"][self.atomic_types[int(fixed_atom[1]/2)]]
            res = solve(matrix, vector)[:-1]
            for fixed_atom in fixed_atoms_index:
                res = insert(res, fixed_atom[0], self.parameters_json["fixed"][self.atomic_types[int(fixed_atom[1]/2)]])
            results[index: new_index] = res
            index = new_index
        self.results = results













#######################################################################################
@jit(nopython=True, cache=True)
def sfkeem_calculate(distances, symbols, all_num_of_atoms, parameters_values):
    results = empty(symbols.size, dtype=float64)
    sigma = parameters_values[-1]
    formal_charge = 0
    index = 0
    counter_distance = 0
    counter_symbols = 0
    for num_of_atoms in all_num_of_atoms:
        new_index = index + num_of_atoms
        num_of_atoms_add_1 = num_of_atoms + 1
        matrix = ones((num_of_atoms_add_1, num_of_atoms_add_1), dtype=float64)
        vector = empty(num_of_atoms_add_1, dtype=float64)
        for x in range(num_of_atoms):
            symbol = symbols[counter_symbols]
            counter_symbols += 1
            vector[x] = - parameters_values[symbol]
            value = parameters_values[symbol + 1]
            for y in range(num_of_atoms):
                matrix[x][y] *= value
                matrix[y][x] *= value
        for x in range(num_of_atoms):
            matrix[x][x] = 2.0 * sqrt(matrix[x][x])
            for y in range(x+1, num_of_atoms):
                matrix[x][y] = matrix[y][x] = 2.0 * sqrt(matrix[x][y]) / cosh(distances[counter_distance] * sigma) # poresit, nejde to rychleji? ale asi nejde...
                counter_distance += 1
        vector[-1] = formal_charge
        matrix[num_of_atoms, num_of_atoms] = 0.0
        results[index: new_index] = solve(matrix, vector)[:-1]
        index = new_index
    return results


class SFKEEM(Methods):
    def calculate(self, set_of_molecules):
        self.results = sfkeem_calculate(set_of_molecules.all_distances, set_of_molecules.multiplied_all_symbolic_numbers_atoms,
                                     set_of_molecules.all_num_of_atoms, self.parameters_values)

##########################################################################################
@jit(nopython=True, cache=True)
def qeq_calculate(distances, all_symbols, all_num_of_atoms, parameters_values):
    results = empty(all_symbols.size, dtype=float64)
    formal_charge = 0
    index = 0
    counter_distance = 0
    for num_of_atoms in all_num_of_atoms:
        new_index = index + num_of_atoms
        symbols = all_symbols[index: new_index]
        num_of_atoms_add_1 = num_of_atoms + 1
        matrix = empty((num_of_atoms_add_1, num_of_atoms_add_1), dtype=float64)
        vector = empty(num_of_atoms_add_1, dtype=float64)
        matrix[num_of_atoms, num_of_atoms] = 0.0
        vector_rad = empty(num_of_atoms, dtype=float64)
        for x in range(num_of_atoms):
            vector_rad[x] = parameters_values[symbols[x] + 2]
            matrix[x][num_of_atoms] = 1.0
            matrix[num_of_atoms][x] = 1.0
        for i in range(num_of_atoms):
            symbol = symbols[i]
            matrix[i][i] = parameters_values[symbol + 1]
            vector[i] = -parameters_values[symbol]
            rad1 = vector_rad[i] # změna!
            for j in range(i + 1, num_of_atoms):
                rad2 = vector_rad[j]
                distance = distances[counter_distance]
                matrix[i][j] = matrix[j][i] = erf(sqrt(rad1 * rad2 / (rad1 + rad2)) * distance) / distance
                counter_distance += 1
        vector[-1] = formal_charge
        results[index: new_index] = solve(matrix, vector)[:-1]
        index = new_index
    return results


class QEq(Methods):
    def calculate(self, set_of_molecules):
        self.results = qeq_calculate(set_of_molecules.all_distances, set_of_molecules.multiplied_all_symbolic_numbers_atoms,
                                     set_of_molecules.all_num_of_atoms, self.parameters_values)


##########################################################################################
@jit(nopython=True, cache=True)
def gm_calculate(all_bonds, all_symbols, all_num_of_atoms, all_num_of_bonds, parameters_values):
    results = empty(all_symbols.size, dtype=float64)
    formal_charge = 0 #???? neni tam
    index_a = 0
    index_b = 0
    for num_of_atoms, num_of_bonds in zip(all_num_of_atoms, all_num_of_bonds):
        new_index_b = index_b + num_of_bonds
        new_index_a = index_a + num_of_atoms
        bonds = all_bonds[index_b: new_index_b]
        symbols = all_symbols[index_a: new_index_a]
        work_electronegativies = zeros(num_of_atoms, dtype=float64)
        work_charges = zeros(num_of_atoms, dtype=float64)
        for alpha in range(4):
            for x in range(num_of_atoms):
                work_charge = work_charges[x]
                parameter_key = symbols[x]
                work_electronegativies[x] = parameters_values[parameter_key] + parameters_values[parameter_key+1] * work_charge + parameters_values[parameter_key+2] * work_charge ** 2
            for bond_index in range(0, len(bonds), 2):
                atom1, atom2 = bonds[bond_index: bond_index+2]
                if work_electronegativies[atom1] < work_electronegativies[atom2]:
                    chi_plus = parameters_values[symbols[atom1]+3]
                else:
                    chi_plus = parameters_values[symbols[atom2]+3]
                charge_diff = ((work_electronegativies[atom1] - work_electronegativies[atom2]) / chi_plus) * 0.5 ** alpha
                work_charges[atom1] -= charge_diff
                work_charges[atom2] += charge_diff
        results[index_a: new_index_a] = work_charges
        index_a = new_index_a
        index_b = new_index_b
    return results


class GM(Methods):
    def calculate(self, set_of_molecules):
        self.results = gm_calculate(set_of_molecules.all_bonds_without_bond_type, set_of_molecules.multiplied_all_symbolic_numbers_atoms,
                                    set_of_molecules.all_num_of_atoms, set_of_molecules.all_num_of_bonds_mul_two,
                                    self.parameters_values)


#################################################################################################
@jit(nopython=True, cache=True)
def mgc_calculate(all_num_of_atoms, all_mgc_matrix, all_symbols, parameters_values):
    results = empty(all_symbols.size, dtype=float64)
    # no formal charge
    index = 0
    counter_symbols = 0
    counter = 0
    for num_of_atoms in all_num_of_atoms:
        new_index = index + num_of_atoms
        vector = empty(num_of_atoms, dtype=float64)
        matrix = empty((num_of_atoms, num_of_atoms), dtype=float64)
        for x in range(num_of_atoms):
            vector[x] = parameters_values[all_symbols[counter_symbols]]
            counter_symbols += 1
            for y in range(x, num_of_atoms):
                matrix[x][y] = matrix[y][x] = all_mgc_matrix[counter]
                counter += 1
        results[index: new_index] = (solve(matrix, vector) - vector)/(prod(vector)**(1/num_of_atoms))
        index = new_index
    return results


class MGC(Methods):
    def calculate(self, set_of_molecules):
        # self.results = mgc_calculate(set_of_molecules.all_num_of_atoms, set_of_molecules.all_MGC_matrix, set_of_molecules.multiplied_all_symbolic_numbers_atoms, self.parameters_values)
        self.results = []
        from time import time
        start = time()
        for molecule in set_of_molecules:
            matrix = zeros((molecule.num_of_atoms, molecule.num_of_atoms), dtype=float64)
            for x in range(molecule.num_of_atoms):
                matrix[x][x] = 1
            for atom1, atom2, bond_type in molecule.bonds_representation("index_index_type"):
                matrix[atom1][atom1] += bond_type
                matrix[atom2][atom2] += bond_type
                matrix[atom1][atom2] -= bond_type
                matrix[atom2][atom1] -= bond_type



            vector = empty(molecule.num_of_atoms, dtype=float64)
            for index, atom in enumerate(molecule.atoms):
                vector[index] = self.parameters[atom.hbo + "_a"]
            r = (solve(matrix, vector) - vector) / (prod(vector) ** (1 / molecule.num_of_atoms))
            for x in r:
                self.results.append(x)
        print(time()- start)
        from sys import exit
        exit()




##############################################################################################
@jit(nopython=True, cache=True)
def acks2_calculate(all_bonds, distances, all_symbols_atoms, all_symbols_bonds, all_num_of_atoms, all_num_of_bonds, parameters_values, num_of_bond_types):
    results = empty(all_symbols_atoms.size, dtype=float64)
    index_b = 0
    index_a = 0
    counter_distance = 0
    bond_parameters_values = parameters_values[-num_of_bond_types:]
    for num_of_atoms, num_of_bonds in zip(all_num_of_atoms, all_num_of_bonds):
        new_index_b = index_b + num_of_bonds
        new_index_a = index_a + num_of_atoms
        symbols_atoms = all_symbols_atoms[index_a: new_index_a]
        symbols_bonds = all_symbols_bonds[int(index_b / 2): int(new_index_b / 2)]
        bonds = all_bonds[index_b: new_index_b]
        matrix = zeros((2 * num_of_atoms + 2, 2 * num_of_atoms + 2))
        vector = zeros(2 * num_of_atoms + 2)
        list_of_q0 = empty(num_of_atoms, dtype=float64)
        list_of_eta = empty(num_of_atoms, dtype=float64)
        for i in range(num_of_atoms):
            symbol = symbols_atoms[i]
            ddd = parameters_values[symbol + 1]
            matrix[i][i] = ddd
            list_of_eta[i] = ddd
            vector[i] = -parameters_values[symbol]
            list_of_q0[i] = parameters_values[symbol + 3]
            for j in range(i+1, num_of_atoms):
                d = distances[counter_distance]
                counter_distance += 1
                d0 = sqrt(2*parameters_values[symbol + 2]**2 + 2*parameters_values[symbols_atoms[j] + 2]**2)
                if d0 == 0:
                    matrix[i,j] = matrix[j,i] = 1.0/d
                else:
                    matrix[i,j] = matrix[j,i] = erf(d/d0)/d
        vector[:num_of_atoms] += list_of_eta * list_of_q0
        matrix[num_of_atoms, :num_of_atoms] = 1
        matrix[:num_of_atoms, num_of_atoms] = 1
        matrix[-1, num_of_atoms + 1:2 * num_of_atoms + 1] = 1
        matrix[num_of_atoms + 1:2 * num_of_atoms + 1, -1] = 1
        vector[num_of_atoms] = sum(list_of_q0)
        vector[num_of_atoms + 1:2 * num_of_atoms + 1] = list_of_q0
        for i in range(num_of_atoms):
            matrix[i, num_of_atoms + 1 + i] = 1.0
            matrix[num_of_atoms + 1 + i, i] = 1.0
        for b_sym, bond_index in zip(symbols_bonds, range(0, len(bonds), 2)):
            atom1, atom2 = bonds[bond_index: bond_index + 2]
            i = num_of_atoms + 1 + atom1
            j = num_of_atoms + 1 + atom2
            bsoft = 1 / bond_parameters_values[b_sym]
            matrix[i, j] += bsoft
            matrix[j, i] += bsoft
            matrix[i, i] -= bsoft
            matrix[j, j] -= bsoft
        results[index_a: new_index_a] = solve(matrix, vector)[:num_of_atoms]
        index_a = new_index_a
        index_b = new_index_b
    return results



class ACKS2(Methods):
    def calculate(self, set_of_molecules):
        self.results = acks2_calculate(set_of_molecules.all_bonds_without_bond_type, set_of_molecules.all_distances, set_of_molecules.multiplied_all_symbolic_numbers_atoms, set_of_molecules.all_symbolic_numbers_bonds,
                                       set_of_molecules.all_num_of_atoms, set_of_molecules.all_num_of_bonds_mul_two, self.parameters_values, len(self.bond_types))





