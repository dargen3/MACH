from .control_existing_files import control_existing_files
from termcolor import colored
from numpy import sqrt, mean
from scipy.stats import pearsonr
from tabulate import tabulate
from .set_of_molecules import SetOfMolecules
from numba import jit

@jit(nopython=True, cache=True)
def calculate_rmsd(rmsd_list):
    count = 0
    for deviation in rmsd_list:
        count += deviation
    return sqrt((1.0/len(rmsd_list))*count)


class Compare:
    def __init__(self, ref_charges_data, charges_data, from_file=False):
        if from_file:
            control_existing_files(((ref_charges_data, True),
                                    (charges_data, True)))
            self.ref_set_of_molecules = SetOfMolecules(ref_charges_data, from_charges_file=True)
            self.set_of_molecules = SetOfMolecules(charges_data, from_charges_file=True)
        else:
            pass
            # zipped_data = zip(ref_charges, charges_data)
        self.statistics()

    def statistics(self):
        print("Calculating statistical data...")
        max_deviation_a = 0
        max_deviation_m = []
        max_deviation_t = {}
        total_deviation_a = 0
        total_deviation_m = []
        total_deviation_t = {}
        rmsd_list_a = []
        rmsd_list_m = []
        rmsd_list_t = {}
        pearson_a = pearsonr([chg for molecule in self.ref_set_of_molecules for chg in molecule.charges],
                               [chg for molecule in self.set_of_molecules for chg in molecule.charges])[0]**2
        pearson_m = []
        pearson_t = {}
        for ref_mol, mol in zip(self.ref_set_of_molecules, self.set_of_molecules):
            rmsd_list_act_m = []
            total_deviation_act_m = 0
            max_deviation_act_m = 0
            pearson_m.append(pearsonr(ref_mol.charges, mol.charges)[0]**2)
            for symbol, ref_at_charge, at_charge in zip(mol.atomic_symbols, ref_mol.charges, mol.charges):
                deviation = abs(at_charge - ref_at_charge)
                deviation_2 = deviation**2

                if max_deviation_a < deviation:
                    max_deviation_a = deviation
                rmsd_list_a.append(deviation_2)
                total_deviation_a += deviation

                if max_deviation_act_m < deviation:
                    max_deviation_act_m = deviation
                rmsd_list_act_m.append(deviation_2)
                total_deviation_act_m += deviation

                if max_deviation_t.setdefault(symbol, 0) < deviation:
                    max_deviation_t[symbol] = deviation
                total_deviation_t.setdefault(symbol, []).append(deviation)
                rmsd_list_t.setdefault(symbol, []).append(deviation_2)
                pearson_t.setdefault(symbol, []).append((ref_at_charge, at_charge))

            rmsd_list_m.append(calculate_rmsd(rmsd_list_act_m))
            total_deviation_m.append(total_deviation_act_m/len(rmsd_list_act_m))
            max_deviation_m.append(max_deviation_act_m)

        average_deviation_a = total_deviation_a/self.set_of_molecules.num_of_atoms
        rmsd_a = calculate_rmsd(rmsd_list_a)
        rmsd_m = mean(rmsd_list_m)

        average_deviation_m = mean(total_deviation_m)
        max_deviation_m = mean(max_deviation_m)
        pearson_m = mean(pearson_m)

        atomic_types_data = []
        for (atomic_type, rmsd), max_deviation, average_deviation, pearson in zip(rmsd_list_t.items(),
                                                                                    max_deviation_t.values(),
                                                                                    total_deviation_t.values(),
                                                                                    pearson_t.values()):
            atomic_types_data.append([atomic_type, calculate_rmsd(rmsd), max_deviation,
                                      mean(average_deviation),
                                      pearsonr([a[0] for a in pearson], [a[1] for a in pearson])[0]**2,
                                      len(average_deviation)])

        print(colored("ok\n", "green"))
        print("Statistics for atoms:\n{}\n".format((tabulate([[rmsd_a, max_deviation_a, average_deviation_a,
                                                                pearson_a, self.set_of_molecules.num_of_atoms]],
                                                              headers=["RMSD", "max deviation", "average deviation",
                                                                       "pearson^2", "num. of atoms"]))))
        print("Statistics for molecules:\n{}\n\n".format((tabulate([[rmsd_m, max_deviation_m, average_deviation_m,
                                                                pearson_m, self.set_of_molecules.num_of_molecules]],
                                                              headers=["RMSD", "max deviation", "average deviation",
                                                                       "pearson^2", "num. of molecules"]))))
        print("Statistics for atomic types:\n{}\n".format(tabulate(atomic_types_data,
                                                                   headers=["atomic type", "RMSD", "max deviation",
                                                                            "average deviation", "pearson^2",
                                                                            "num. of atoms"])))
