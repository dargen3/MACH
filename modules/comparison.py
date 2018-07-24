from .control_existing_files import control_existing_files
from termcolor import colored
from numpy import sqrt, mean, max, sum, corrcoef
from scipy.stats import pearsonr
from tabulate import tabulate
from .set_of_molecules import SetOfMolecules


def calculate_statistics(ref_charges, charges):
    deviations = abs(ref_charges - charges)
    rmsd = sqrt((1.0/deviations.size)*sum(deviations**2))
    max_deviation = max(deviations)
    average_deviation = mean(deviations)
    pearson_2 = corrcoef(ref_charges, charges)[0, 1] ** 2
    return [rmsd, max_deviation, average_deviation, pearson_2, deviations.size]


class Compare:
    def __init__(self, ref_charges_data, charges_data,  save_fig, from_file=False):
        if from_file:
            control_existing_files(((ref_charges_data, True),
                                    (charges_data, True)))
            self.ref_set_of_molecules = SetOfMolecules(ref_charges_data, from_charges_file=True)
            self.set_of_molecules = SetOfMolecules(charges_data, from_charges_file=True)
        else:
            pass
            # zipped_data = zip(ref_charges, charges_data)
        self.statistics()
        if save_fig:
            self.graphs()

    def graphs(self):
        from matplotlib import pyplot as plt, cm
        from sys import exit
        all_atoms_graph = plt.figure(figsize=(11, 9)).add_subplot(111)
        colors = cm.tab20.colors
        color_numbers = {"H~1": 0, "O~1": 2, "O~2": 4, "N~1": 6, "N~2": 8, "C~1": 10, "C~2": 12, "S~1": 14,
                         "Ca~1": 16, "S~2": 18, "P~1": 1, "P~2": 3, "N~3": 5, "C~3": 7, "Br~1": 9, "Cl~1": 9,
                         "F~1": 9, "I~1": 9, "H": 0, "C": 2, "N": 4, "O": 6, "S": 8}

        for atomic_symbol, (ref_charges, charges) in self.atomic_types_data.items():
            try:
                color_number = color_numbers[atomic_symbol]
            except KeyError:
                exit("No color for {} atomic type!".format(atomic_symbol))
            all_atoms_graph.scatter(ref_charges, charges, marker=".", color=colors[color_number], label=atomic_symbol)
            atom_type_graph = plt.figure(figsize=(11, 9)).add_subplot(111, rasterized=True)
            atom_type_graph.set_title(atomic_symbol)
            atom_type_graph.set_xlabel(self.ref_set_of_molecules.file, fontsize=15)
            atom_type_graph.set_ylabel(self.set_of_molecules.file, fontsize=15)
            for symbol, (ref_chg, chg) in self.atomic_types_data.items():
                if symbol == atomic_symbol:
                    continue
                atom_type_graph.scatter(ref_chg, chg, marker=".", color="gainsboro")
            atom_type_graph.scatter(ref_charges, charges, marker=".", color="black")

        # plt.text(axis_range[1], axis_range[0], "Num. of atoms: " + str(len(list_with_charges1)) + "\nrmsd: " +
            #          str(rmsd)[:6] + "\nPearson**2: " + str(person_2)[:6], ha='right', va='bottom', fontsize=15)
            # name = "{}-{}".format(name_of_all_set, atomic_symbol)
            # plt.savefig(name, dpi=300)
        all_atoms_graph.legend(fontsize=15)

        plt.show()



    def statistics(self):
        print("Calculating statistical data...")
        atomic_types_statistical_data = []
        for (atomic_type, ref_charges), (_, charges) in zip(self.ref_set_of_molecules.atomic_types_charges.items(),
                                                            self.set_of_molecules.atomic_types_charges.items()):
            atomic_types_statistical_data.append([atomic_type] + calculate_statistics(ref_charges, charges))
        molecules_statistical_data = []
        for ref_molecule, molecule in zip(self.ref_set_of_molecules, self.set_of_molecules):
            data.append(calculate_statistics(ref_molecule.charges, molecule.charges))
        print(colored("ok\n", "green"))
        print("Statistics for atoms:\n{}\n".format((tabulate([calculate_statistics(self.ref_set_of_molecules.all_charges,
                                                                                   self.set_of_molecules.all_charges)],
                                                              headers=["RMSD", "max deviation", "average deviation",
                                                                       "pearson^2", "num. of atoms"]))))
        print("Statistics for molecules:\n{}\n\n".format((tabulate([[mean([x[0] for x in molecules_statistical_data]),
                                                                     mean([x[1] for x in molecules_statistical_data]),
                                                                     mean([x[2] for x in molecules_statistical_data]),
                                                                     mean([x[3] for x in molecules_statistical_data]),
                                                                     self.set_of_molecules.num_of_molecules]],
                                                              headers=["RMSD", "max deviation", "average deviation",
                                                                       "pearson^2", "num. of molecules"]))))
        print("Statistics for atomic types:\n{}\n".format(tabulate(atomic_types_statistical_data,
                                                                   headers=["atomic type", "RMSD", "max deviation",
                                                                            "average deviation", "pearson^2",
                                                                            "num. of atoms"])))
