from .set_of_molecules import SetOfMolecules
from .submolecules import create_submolecules
from .control_existing import control_existing_files
from .output_files import write_charges_to_file
from importlib import import_module
from termcolor import colored
from numpy import linalg
from os.path import basename


class Calculation:
    def __init__(self, sdf, method, parameters, charges, atomic_types_pattern, submolecules, rewriting_with_force):
        original_sdf = sdf
        files = [(sdf, True, "file"),
                 (charges, False, "file")]
        if submolecules:
            files.extend([(basename(sdf)+".submolecules", False, "file")])
        control_existing_files(files, rewriting_with_force)
        if submolecules:
            submolecules, sdf = create_submolecules(sdf, cutoff=12)
        method = getattr(import_module("modules.methods"), method)()
        set_of_molecules = SetOfMolecules(sdf, submolecules=submolecules)
        method.load_parameters(parameters, set_of_molecules, "calculation", atomic_types_pattern)
        set_of_molecules.create_method_data(method)
        print("Calculation of charges... ")
        try:
            method.calculate(set_of_molecules)
        except (linalg.linalg.LinAlgError, ZeroDivisionError) as e:
            print(e)
        print(colored("ok\n", "green"))
        if submolecules:
            set_of_molecules = SetOfMolecules(original_sdf)
            set_of_molecules.create_method_data(method)
        write_charges_to_file(charges, method.results, set_of_molecules)
