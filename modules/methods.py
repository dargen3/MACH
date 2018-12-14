from sys import exit
from termcolor import colored
from numba import jit
from numpy import float64, empty, array, ones, zeros, sqrt, cosh, concatenate, int64, sum, prod, dot
from numpy.linalg import solve, eigvalsh
from math import erf
from json import load

class Methods:
    def __init__(self):
        self.necessarily_data = {"EEM": ["distances"],
                                 "QEq": ["distances"],
                                 "SFKEEM": ["distances"],
                                 "GM": ["bonds_without_bond_type", "num_of_bonds_mul_two"],
                                 "MGC": ["MGC_matrix"],
                                 "SQE": ["distances", "num_of_bonds_mul_two", "bonds_without_bond_type"],
                                 "ACKS2": ["distances", "num_of_bonds_mul_two", "bonds_without_bond_type"],
                                 "DDACEM": ["distances"]
                                 }[str(self)]

    def control_parameters(self, file, all_symbolic_numbers_atoms):
        missing_atoms_in_parameters = [self.atomic_types[sym_num] for sym_num in range(len(self.atomic_types)) if sym_num not in all_symbolic_numbers_atoms]
        if missing_atoms_in_parameters:
            exit(colored("No {} atoms in {}. Parameterization is not possible.".format(", ".join(missing_atoms_in_parameters), file), "red"))

    def convert_atom(self, atom):
        if isinstance(atom, list):
            if atom[1] == "hbo":
                return "{}~{}".format(atom[0], atom[2])
            elif atom[1] == "plain":
                return atom[0]
        elif isinstance(atom, str):
            s_atom = atom.split("~")
            if len(s_atom) == 2:
                return [s_atom[0], "hbo", s_atom[1]]
            elif len(s_atom) == 1:
                return [atom, "plain", "*"]

    def convert_bond(self, bond):
        if isinstance(bond, list):
            if bond[2] == "hbo":
                return "bond-{}~{}".format("-".join(*sorted([bond[:2]])), bond[3])
            elif bond[2] == "plain":
                return "bond-{}-{}".format(*sorted([bond[:2]]))
        elif isinstance(bond, str):
            s_bond = bond.split("-")
            if "~" in bond:
                return [s_bond[1], s_bond[2].split("~")[0], "hbo", bond[-1]]
            else:
                return [*s_bond[1:3], "plain" ,"*"]

    def __repr__(self):
        return self.__class__.__name__

    def load_parameters(self, parameters_file):
        if not parameters_file:
            parameters_file = "modules/parameters/{}.par".format(str(self))
        print("Loading of parameters from {}...".format(parameters_file))
        self.parameters_json = load(open(parameters_file))
        method_in_parameters_file = self.parameters_json["metadata"]["method"]
        if self.__class__.__name__ != method_in_parameters_file:
            exit(colored("These parameters are for method {} but you want to calculate charges by method {}!\n"
                         .format(method_in_parameters_file, self.__class__.__name__), "red"))
        self.parameters = {}
        if "common" in self.parameters_json:
            print("\n\ntest\n\n")
            for name, value in zip(self.parameters_json["common"]["names"], self.parameters_json["common"]["values"]):
                self.parameters[name] = value
        self.atomic_parameters_types = self.parameters_json["atom"]["names"]
        atomic_types = []
        for parameter in self.parameters_json["atom"]["data"]:
            atomic_type = self.convert_atom(parameter["key"])
            atomic_types.append(atomic_type)
            for parameter_name, value in zip(self.atomic_parameters_types, parameter["value"]):
                self.parameters["{}_{}".format(atomic_type, parameter_name)] = value
        self.atomic_types = sorted(atomic_types)
        if "bond" in self.parameters_json:
            for parameter in self.parameters_json["bond"]["data"]:
                self.parameters[self.convert_bond(parameter["key"])] = parameter["value"][0]
        self.atomic_types_pattern = "atomic_symbol_high_bond" if self.parameters_json["atom"]["data"][0]["key"][1] == "hbo" else "atomic_symbol"
        self.bond_types = sorted([key for key in self.parameters.keys() if key[:5] == "bond-"])
        writed_glob_par = -1
        num_of_atom_par = len(self.atomic_parameters_types) * len(self.atomic_types)
        parameters_values = [0 for _ in range(len(self.parameters))]
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
                atomic_type["value"][index] = self.parameters["{}_{}".format(self.convert_atom(atomic_type["key"]), parameter)]
        if "bond" in self.parameters_json:
            for bond in self.parameters_json["bond"]["data"]:
                bond["value"] = self.parameters[self.convert_bond(bond["key"])]


#####################################################################

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
        # mmm = max(abs(rrr))
        # if mmm > 10:
        #     print(mmm, cond(matrix))
        index = new_index
    return results


class EEM(Methods):
    def calculate(self, set_of_molecules):
        self.results = eem_calculate(set_of_molecules.all_distances, set_of_molecules.multiplied_all_symbolic_numbers_atoms,
                                     set_of_molecules.all_num_of_atoms, self.parameters_values)

##########################################################################################
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
            for j in range(i + 1, num_of_atoms):
                rad1 = vector_rad[i]
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
        self.results = mgc_calculate(set_of_molecules.all_num_of_atoms, set_of_molecules.all_MGC_matrix, set_of_molecules.multiplied_all_symbolic_numbers_atoms, self.parameters_values)


#####################################################################################
@jit(nopython=True)
def sqe_calculate(all_bonds, distances, all_symbols_atoms, all_symbols_bonds, all_num_of_atoms, all_num_of_bonds, parameters_values, num_of_bond_types):
    results = empty(all_symbols_atoms.size, dtype=float64)
    index_b = 0
    index_a = 0
    counter_distance = 0
    bond_parameters_values = parameters_values[-num_of_bond_types:]
    for num_of_atoms, num_of_bonds in zip(all_num_of_atoms, all_num_of_bonds):
        new_index_b = index_b + num_of_bonds
        new_index_a = index_a + num_of_atoms
        symbols_atoms = all_symbols_atoms[index_a: new_index_a]
        symbols_bonds = all_symbols_bonds[int(index_b/2): int(new_index_b/2)]
        T = zeros((int(num_of_bonds/2), num_of_atoms))
        bonds = all_bonds[index_b: new_index_b]
        for index, bond_index in enumerate(range(0, len(bonds), 2)):
            atom1, atom2 = bonds[bond_index: bond_index + 2]
            T[index, atom1] += 1
            T[index, atom2] -= 1
        matrix = zeros((num_of_atoms, num_of_atoms))
        vector = zeros(num_of_atoms)
        list_of_q0 = empty(num_of_atoms, dtype=float64)
        list_of_eta = empty(num_of_atoms, dtype=float64)
        for i in range(num_of_atoms):
            symbol = symbols_atoms[i]
            matrix[i][i] = parameters_values[symbol + 1]
            list_of_eta[i] = parameters_values[symbol + 1]
            vector[i] = -parameters_values[symbol]
            list_of_q0[i] = parameters_values[symbol + 3]
            for j in range(i+1, num_of_atoms):
                d = distances[counter_distance]
                counter_distance += 1
                d0 = sqrt(2*parameters_values[symbol + 2]**2 + 2*parameters_values[symbols_atoms[j] +2]**2)
                if d0 == 0:
                    matrix[i,j] = matrix[j,i] = 1.0/d
                else:
                    matrix[i,j] = matrix[j,i] = erf(d/d0)/d
        vector -= dot(matrix, list_of_q0)
        vector += list_of_eta*list_of_q0
        A_sqe = dot(T, dot(matrix, T.T))
        B_sqe = dot(T, vector)
        for index, b_sym in enumerate(symbols_bonds):
            A_sqe[index, index] += bond_parameters_values[b_sym]
        results[index_a: new_index_a] = dot(solve(A_sqe, B_sqe), T) + list_of_q0
        index_a = new_index_a
        index_b = new_index_b
    return results


class SQE(Methods):
    def calculate(self, set_of_molecules):
        self.results = sqe_calculate(set_of_molecules.all_bonds_without_bond_type, set_of_molecules.all_distances, set_of_molecules.multiplied_all_symbolic_numbers_atoms, set_of_molecules.all_symbolic_numbers_bonds,
                       set_of_molecules.all_num_of_atoms, set_of_molecules.all_num_of_bonds_mul_two, self.parameters_values, len(self.bond_types))


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
            matrix[i][i] = parameters_values[symbol + 1]
            list_of_eta[i] = parameters_values[symbol + 1]
            vector[i] = -parameters_values[symbol]
            list_of_q0[i] = parameters_values[symbol + 3]
            for j in range(i+1, num_of_atoms):
                d = distances[counter_distance]
                counter_distance += 1
                d0 = sqrt(2*parameters_values[symbol + 2]**2 + 2*parameters_values[symbols_atoms[j] +2]**2)
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










@jit(nopython=True)
def ddacem_calculate(all_distances, all_symbols, all_num_of_atoms, parameters_values):
    results = empty(all_symbols.size, dtype=float64)
    index = 0
    index_dist = 0
    gchg = parameters_values[-2]
    gel = parameters_values[-1]
    for num_of_atoms in all_num_of_atoms:
        new_index = index + num_of_atoms
        new_index_dist = int(index_dist + num_of_atoms * (num_of_atoms - 1) /2)
        symbols = all_symbols[index: new_index]
        distances = all_distances[index_dist: new_index_dist]
        DDAEs = zeros(num_of_atoms, dtype=float64)
        actual_distance_index = 0
        for j in range(num_of_atoms):
            for i in range(j+1, num_of_atoms):
                distance = distances[actual_distance_index]
                actual_distance_index += 1
                if distance < 2:
                    DDAE = ((parameters_values[symbols[j]] - parameters_values[symbols[i]]) / distance)
                    DDAEs[j] += DDAE
                    DDAEs[i] -= DDAE
        vector = zeros(num_of_atoms+1, dtype=float64)
        for x in range(num_of_atoms):
            vector[x] = parameters_values[symbols[x]+1] + DDAEs[x] * parameters_values[symbols[x]+2] + DDAEs[x] * gel
        matrix = zeros((num_of_atoms+1, num_of_atoms+1), dtype=float64)
        actual_distance_index = 0
        for x in range(num_of_atoms):
            matrix[x][x] += 1
            matrix[x][num_of_atoms] = matrix[num_of_atoms][x] = 1
            for y in range(x+1, num_of_atoms):
                distance = distances[actual_distance_index]
                if distance < 5:
                    matrix[x][x] -= gchg/(distance**2)
                    matrix[y][y] -= gchg/(distance**2)
                    matrix[x][y] += gchg/(distance**2)
                    matrix[y][x] += gchg/(distance**2)
                actual_distance_index += 1
        matrix[num_of_atoms][num_of_atoms] = 1
        charges = solve(matrix, vector)[:-1]
        results[index: new_index] = charges
        index = new_index
        index_dist = new_index_dist
    return results


class DDACEM(Methods):
    def calculate(self, set_of_molecules):
        self.results = ddacem_calculate(set_of_molecules.all_distances, set_of_molecules.multiplied_all_symbolic_numbers_atoms,
                                     set_of_molecules.all_num_of_atoms, self.parameters_values)
