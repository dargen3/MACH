from .set_of_molecules import SetOfMolecules
from .control_existing import control_existing_files
from importlib import import_module
from termcolor import colored


def write_charges_to_file(charges, results, set_of_molecules):
    print("Writing charges to {}...".format(charges))
    with open(charges, "w") as file_with_results:
        count = 0
        for molecule in set_of_molecules:
            file_with_results.write("{}\n{}\n".format(molecule.name, molecule.num_of_atoms))
            for index, atom in enumerate(molecule.atomic_symbols):
                file_with_results.write("{0:>3} {1:>3} {2:>15}\n".format(index + 1, atom, str(float("{0:.6f}".format(results[count])))))
                count += 1
            file_with_results.write("\n")
    print(colored("ok\n", "green"))


class Calculation:
    def __init__(self, sdf, method, parameters, charges, rewriting_with_force):
        control_existing_files(((sdf, True, "file"),
                                (parameters, True, "file"),
                                (charges, False, "file")),
                               rewriting_with_force)
        method = getattr(import_module("modules.methods"), method)()
        method.load_parameters(parameters)
        set_of_molecules = SetOfMolecules(sdf)
        method.create_method_data(set_of_molecules)
        print("Calculation of charges... ")
        method.load_array_for_results(set_of_molecules.num_of_atoms)
        method.calculate(set_of_molecules)
        print(colored("ok\n", "green"))
        write_charges_to_file(charges, method.results, set_of_molecules)
