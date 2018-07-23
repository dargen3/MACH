from sys import exit
from termcolor import colored
from numba import jit
import numpy as np



class Methods:
    def load_parameters(self, parameters_file):
        print("Loading of parameters from {}...".format(parameters_file))
        with open(parameters_file, "r") as parameters_file:
            parameters_file = parameters_file.read()
            for keyword in ["<<global>>", "<<key>>", "<<value_symbol>>", "<<value>>", "<<end>>"]:
                if keyword not in parameters_file:
                    exit(colored("File with parameters is wrong! File not contain keywords <<global>>, <<key>>, "
                                 "<<value_symbol>>, <<value>> and <<end>>!\n", "red"))
            parameters_file = parameters_file.splitlines()
            method_in_parameters_file = parameters_file[0].split()[1]

            if self.__class__.__name__ not in [None, method_in_parameters_file]:
                exit(colored("These parameters are for method {} but you want to calculate charges by method {}!\n"
                             .format(method_in_parameters_file, method.name), "red"))
            length_corrections = {"Angstrom":1.0, "Bohr_radius":1.8897261245}
            try:
                length_correction = parameters_file[1].split()[1]
                self.length_correction = length_corrections[length_correction]
            except KeyError:
                exit(colored("Unknown length type {} in parameter file. Please, choice one of {}.\n"
                             .format(length_correction, ", ".join(length_corrections.keys())), "red"))
            key_index = parameters_file.index("<<key>>")
            value_symbol_index = parameters_file.index("<<value_symbol>>")
            value_index = parameters_file.index("<<value>>")
            end_index = parameters_file.index("<<end>>")
            self.parameters = {}
            for line in parameters_file[3: key_index]:
                key, value = line.split()
                self.parameters[key] = float(value)
            keys = [line for line in parameters_file[key_index + 1: value_symbol_index]]
            self.atomic_types_pattern = "_".join([key for key in keys])
            value_symbols = [line for line in parameters_file[value_symbol_index + 1: value_index]]
            self.atomic_types = []
            for line in parameters_file[value_index + 1: end_index]:
                atomic_type_data = line.split()
                for x in range(len(value_symbols)):
                    atomic_type = "~".join(atomic_type_data[:len(keys)])
                    self.atomic_types.append(atomic_type)
                    self.parameters["{}_{}".format(atomic_type, value_symbols[x])] = float(atomic_type_data[x + len(keys)])
            self.atomic_types = sorted(tuple(set(self.atomic_types)))
            self.parameters_numbers = [[] for x in range(len(self.atomic_types))]
            for parameter in self.parameters:
                if parameter[0].isupper():
                    self.parameters_numbers[self.atomic_types.index(parameter.split("_")[0])].append(self.parameters[parameter])
            self.parameters_numbers = np.array(self.parameters_numbers, dtype=np.float64)
        print(colored("ok\n", "green"))

    def load_array_for_results(self, num_of_atoms):
        self.results = np.empty(num_of_atoms)

@jit(nopython=True, cache=True)
def fill_array(array, data, index):
    count = 0
    for x in data:
        array[index + count] = x
        count += 1
    return count

@jit(nopython=True, cache=True)
def eem_calculate(num_of_atoms, kappa, matrix_of_distance, parameters_values, parameters_keys, formal_charge):
    matrix = np.empty((num_of_atoms + 1, num_of_atoms + 1), dtype=np.float64)
    vector = np.empty(num_of_atoms + 1, dtype=np.float64)
    matrix[:num_of_atoms, :num_of_atoms] = kappa / matrix_of_distance
    matrix[num_of_atoms, :] = 1.0
    matrix[:, num_of_atoms] = 1.0
    matrix[num_of_atoms, num_of_atoms] = 0.0
    for i in range(num_of_atoms):
        symbol = parameters_keys[i]
        matrix[i][i] = parameters_values[symbol][1]
        vector[i] = -parameters_values[symbol][0]
    vector[-1] = formal_charge
    results = np.linalg.solve(matrix, vector)
    return results[:-1]


class EEM(Methods):
    def calculate(self, set_of_molecules):
        index = 0
        for molecule in set_of_molecules:
            index += fill_array(self.results, eem_calculate(len(molecule), self.parameters["kappa"], molecule.distance_matrix,
                       self.parameters_numbers, molecule.symbolic_numbers, 0), index)


    def calculate_slow(self, molecule):
        num_of_atoms = len(molecule)
        matrix = np.empty((num_of_atoms + 1, num_of_atoms + 1), dtype=float)
        vector = np.empty(shape=[num_of_atoms + 1], dtype=float)
        matrix[:num_of_atoms, :num_of_atoms] = self.get_parameter("kappa") / molecule.matrix_of_distance
        matrix[num_of_atoms, :] = 1.0
        matrix[:, num_of_atoms] = 1.0
        matrix[num_of_atoms, num_of_atoms] = 0.0
        for i in range(1, num_of_atoms + 1):
            symbol = self.symbol(i, molecule)
            i_min1 = i - 1
            matrix[i_min1][i_min1] = self.get_parameter(symbol + "~beta")
            vector[i_min1] = -self.get_parameter(symbol + "~alfa")
        vector[-1] = molecule.formal_charge
        results = np.linalg.solve(matrix, vector)
        return results[:-1]

class SFKEEM(Methods):
    pass