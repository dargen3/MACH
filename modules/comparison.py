from .control_existing_files import control_existing_files
from termcolor import colored
from numpy import sqrt, mean, max, min, sum, corrcoef
from scipy.stats import pearsonr
from tabulate import tabulate
from .set_of_molecules import SetOfMolecules
from matplotlib import pyplot as plt, cm
from sys import exit
from collections import namedtuple


def calculate_statistics(ref_charges, charges):
    deviations = abs(ref_charges - charges)
    rmsd = sqrt((1.0/deviations.size)*sum(deviations**2))
    max_deviation = max(deviations)
    average_deviation = mean(deviations)
    pearson_2 = corrcoef(ref_charges, charges)[0, 1] ** 2
    return [rmsd, max_deviation, average_deviation, pearson_2, deviations.size]


class Comparison:
    def __init__(self, ref_charges_data, charges_data,  save_fig, from_file=False):
        if from_file:
            control_existing_files(((ref_charges_data, True),
                                    (charges_data, True)))
            self.ref_set_of_molecules = SetOfMolecules(ref_charges_data, from_charges_file=True)
            self.set_of_molecules = SetOfMolecules(charges_data, from_charges_file=True)
        else:
            set_of_molecules_nt = namedtuple("set_of_molecules", ("all_charges", "atomic_types_charges", "molecules"))
            self.ref_set_of_molecules = ref_charges_data
            self.set_of_molecules = set_of_molecules_nt(charges_data[0], charges_data[1], charges_data[2])
        self.statistics()
        if save_fig:
            self.graphs()

    def graphs(self):
        fig = plt.figure(figsize=(11, 9))
        all_atoms_graph = fig.add_subplot(111)
        colors = cm.tab20.colors
        color_numbers = {"H~1": 0, "O~1": 2, "O~2": 4, "N~1": 6, "N~2": 8, "C~1": 10, "C~2": 12, "S~1": 14,
                         "Ca~1": 16, "S~2": 18, "P~1": 1, "P~2": 3, "N~3": 5, "C~3": 7, "Br~1": 9, "Cl~1": 9,
                         "F~1": 9, "I~1": 9, "H": 0, "C": 2, "N": 4, "O": 6, "S": 8}
        zipped_charges = list(zip(self.ref_set_of_molecules.atomic_types_charges.items(),
                                  self.set_of_molecules.atomic_types_charges.items()))
        for index, ((atomic_symbol, ref_charges), (_, charges)) in enumerate(zipped_charges):
            try:
                color_number = color_numbers[atomic_symbol]
            except KeyError:
                exit(colored("No color for {} atomic type!".format(atomic_symbol), "red"))
            all_atoms_graph.scatter(ref_charges, charges, marker=".", color=colors[color_number], label=atomic_symbol)
            atom_type_graph = plt.figure(figsize=(11, 9)).add_subplot(111, rasterized=True)
            atom_type_graph.set_title(atomic_symbol)
            atom_type_graph.set_xlabel(self.ref_set_of_molecules.file, fontsize=20)
            atom_type_graph.set_ylabel(self.set_of_molecules.file, fontsize=20)
            for (symbol, ref_chg), (_, chg) in zipped_charges:
                if symbol == atomic_symbol:
                    continue
                atom_type_graph.scatter(ref_chg, chg, marker=".", color="gainsboro")
            atom_type_graph.scatter(ref_charges, charges, marker=".", color="black")
            plt.text(plt.xlim()[1] - 0.1, plt.ylim()[0] + 0.1, "RMSD: {:.3f}\n$R^2$: {:.3f}".
                     format(self.atomic_types_statistical_data[index][1],
                            self.atomic_types_statistical_data[index][4]), ha='right', va='bottom', fontsize=28)
            plt.savefig(atomic_symbol, dpi=300)
        all_atoms_graph.text(plt.xlim()[1] - 0.1, plt.ylim()[0] + 0.1, "RMSD: {:.3f}\n$R^2$: {:.3f}".
                             format(self.all_atoms_data[0], self.all_atoms_data[3]), ha='right', va='bottom', fontsize=28)
        all_atoms_graph.legend(fontsize=14, loc="upper left")
        fig.savefig("all_atomic_types.png", dpi=300)
        plt.show()

    def statistics(self):
        print("Calculating statistical data...")
        self.all_atoms_data = calculate_statistics(self.ref_set_of_molecules.all_charges,
                                                   self.set_of_molecules.all_charges)
        self.atomic_types_statistical_data = []
        for (atomic_type, ref_charges), (_, charges) in zip(self.ref_set_of_molecules.atomic_types_charges.items(),
                                                            self.set_of_molecules.atomic_types_charges.items()):
            self.atomic_types_statistical_data.append([atomic_type] + calculate_statistics(ref_charges, charges))
        molecules_statistical_data = []
        for ref_molecule, molecule in zip(self.ref_set_of_molecules, self.set_of_molecules.molecules):
            molecules_statistical_data.append(calculate_statistics(ref_molecule.charges, molecule.charges))
        print(colored("ok\n", "green"))
        print("Statistics for atoms:\n{}\n".format((tabulate([self.all_atoms_data],
                                                              headers=["RMSD", "max deviation", "average deviation",
                                                                       "pearson^2", "num. of atoms"]))))
        print("Statistics for molecules:\n{}\n\n".format((tabulate([[mean([x[0] for x in molecules_statistical_data]),
                                                                     mean([x[1] for x in molecules_statistical_data]),
                                                                     mean([x[2] for x in molecules_statistical_data]),
                                                                     mean([x[3] for x in molecules_statistical_data]),
                                                                     self.ref_set_of_molecules.num_of_molecules]],
                                                              headers=["RMSD", "max deviation", "average deviation",
                                                                       "pearson^2", "num. of molecules"]))))
        print("Statistics for atomic types:\n{}\n".format(tabulate(self.atomic_types_statistical_data,
                                                                   headers=["atomic type", "RMSD", "max deviation",
                                                                            "average deviation", "pearson^2",
                                                                            "num. of atoms"])))
