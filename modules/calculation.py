from .set_of_molecules import SetOfMolecules
from .control_existing import control_existing_files
from .output_files import write_charges_to_file
from importlib import import_module
from termcolor import colored
from numpy import linalg
from os.path import basename


class Calculation:
    def __init__(self, sdf, method, parameters, charges, atomic_types_pattern, rewriting_with_force):
        control_existing_files([(sdf, True, "file"),
                                (charges, False, "file")], rewriting_with_force)
        method = getattr(import_module("modules.methods"), method)()
        set_of_molecules = SetOfMolecules(sdf)
        method.load_parameters(parameters, set_of_molecules, "calculation", atomic_types_pattern)
        set_of_molecules.create_method_data(method)
        print("Calculation of charges... ")
        try:
            method.calculate(set_of_molecules)
        except (linalg.linalg.LinAlgError, ZeroDivisionError) as e:
            print(e)
        print(colored("ok\n", "green"))
        write_charges_to_file(charges, method.results, set_of_molecules)
