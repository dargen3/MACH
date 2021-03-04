from importlib import import_module

from numpy import linalg
from termcolor import colored

from .control_input import control_and_copy_input_files
from .set_of_molecules import create_set_of_mols, create_method_data, write_chgs_to_file


def calculate_chgs(sdf_file: str,
                   chg_method: str,
                   params_file: str,
                   data_dir: str):

    control_and_copy_input_files(data_dir,
                                 (sdf_file, params_file))

    chg_method = getattr(import_module(f"modules.chg_methods.{chg_method}"), chg_method)()
    ats_types_pattern = chg_method.load_params(params_file)
    set_of_mols = create_set_of_mols(sdf_file, ats_types_pattern)
    set_of_mols.emp_chgs_file = f"{data_dir}/output_files/empirical.chg"
    chg_method.prepare_params_for_calc(set_of_mols)

    print("Preprocessing data...")
    create_method_data(chg_method, set_of_mols)
    print(colored("ok\n", "green"))

    print("Calculation of charges... ")
    try:
        emp_chgs = chg_method.calculate(set_of_mols)
    except (linalg.linalg.LinAlgError, ZeroDivisionError) as error:
        print(error)
    print(colored("ok\n", "green"))

    write_chgs_to_file(emp_chgs, set_of_mols)
