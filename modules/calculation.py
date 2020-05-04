from importlib import import_module
from json import load

from numpy import linalg
from termcolor import colored

from .control_existing import control_existing_files
from .input_output import write_charges_to_file
from .set_of_molecules import create_set_of_molecules, create_method_data


def calculate_chg(sdf_file: str,
                  par_file: str,
                  chg_method: str,
                  emp_chg_file: str,
                  rewriting_with_force: bool):

    control_existing_files([(sdf_file, True, "file"),
                            (par_file, True, "file"),
                            (emp_chg_file, False, "file")],
                           rewriting_with_force)

    atomic_types_pattern = load(open(par_file))["metadata"]["atomic_types_pattern"]
    chg_method = getattr(import_module("modules.methods"), chg_method)()
    set_of_molecules = create_set_of_molecules(sdf_file, atomic_types_pattern)
    chg_method.load_parameters(par_file, set_of_molecules, "calculation", atomic_types_pattern)

    print("Preprocessing data...")
    create_method_data(chg_method, set_of_molecules)
    print(colored("ok\n", "green"))

    print("Calculation of charges... ")
    try:
        chg_method.calculate(set_of_molecules)
    except (linalg.linalg.LinAlgError, ZeroDivisionError) as error:
        print(error)
    print(colored("ok\n", "green"))

    write_charges_to_file(emp_chg_file, chg_method.results, set_of_molecules)
