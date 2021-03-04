from collections import namedtuple
from datetime import datetime as date
from heapq import nsmallest
from importlib import import_module
from sys import argv

import numpy as np
from termcolor import colored

from .chg_methods.chg_method import ChargeMethod
from .comparison import comparison_par
from .control_input import control_and_copy_input_files
from .set_of_molecules import *
from .optimization_methods.local_minimization import local_minimization
from .optimization_methods.GDMIN import GDMIN
from .optimization_methods.optGM import optGM

global obj_eval
global obj_eval_error
obj_eval = 0
obj_eval_error = 0

















def objective_function(params: np.array,
                       chg_method: ChargeMethod,
                       set_of_mols: SetOfMolecules,
                       obj_vals: list = None) -> float:

    global obj_eval, obj_eval_error

    def _rmsd_calculation(emp_chg: np.array) -> tuple:

        ats_types_rmsd = np.empty(len(chg_method.ats_types))
        for index, symbol in enumerate(chg_method.ats_types):
            differences = emp_chg[set_of_mols.all_ats_ids == index * chg_method.params_per_at_type] - set_of_mols.ref_ats_types_chgs[symbol]
            ats_types_rmsd[index] = np.sqrt(np.mean(np.abs(differences) ** 2))
        total_mols_rmsd = 0
        index = 0
        for mol in set_of_mols.mols:
            new_index = index + mol.num_of_ats
            differences = emp_chg[index: new_index] - mol.ref_chgs
            total_mols_rmsd += np.sqrt(np.mean(np.abs(differences) ** 2))
            index = new_index
        return ats_types_rmsd, total_mols_rmsd / set_of_mols.num_of_mols

    obj_eval += 1
    chg_method.params_vals = params
    try:
        emp_chgs = chg_method.calculate(set_of_mols)
    except (np.linalg.LinAlgError, ZeroDivisionError) as e:
        obj_eval_error += 1
        print(e)
        return 1000.0
    ats_types_rmsd, rmsd = _rmsd_calculation(emp_chgs)
    objective_val = rmsd + np.mean(ats_types_rmsd)
    if np.isnan(objective_val):
        return 1000.0
    if isinstance(obj_vals, list):
        obj_vals.append(objective_val)
    print("    Total RMSD: {}    Worst RMSD: {}".format(str(rmsd)[:8], str(np.max(ats_types_rmsd))[:8]), end="\r")
    return objective_val


def parameterize(sdf_file: str,
                 ref_chgs_file: str,
                 chg_method: str,
                 params_file: str,
                 ats_types_pattern: str,
                 percent_par: int,
                 optimization_method: str,
                 num_of_samples: int,
                 num_of_candidates: int,
                 subset: int,
                 min_subset: int,
                 maxiter: int,
                 seed: int,
                 data_dir: str):

    start_time = date.now()
    control_and_copy_input_files(data_dir,
                                 (file for file in (sdf_file, ref_chgs_file, params_file) if file))

    chg_method = getattr(import_module(f"modules.chg_methods.{chg_method}"), chg_method)()
    ats_types_pattern = chg_method.load_params(params_file, ats_types_pattern)
    set_of_mols = create_set_of_mols(sdf_file, ats_types_pattern)
    set_of_mols.emp_chgs_file = f"{data_dir}/output_files/empirical.chg"
    chg_method.prepare_params_for_par(set_of_mols)
    add_chgs(set_of_mols, ref_chgs_file, "ref_chgs")

    set_of_mols_par, set_of_mols_val = create_par_val_set(set_of_mols, percent_par, seed)
    if optimization_method == "optGM":
        subset_of_mols, min_subset_of_mols = create_subset_minsubset(set_of_mols_par, subset, min_subset)

    print("Preprocessing data...")
    create_method_data(chg_method, set_of_mols)
    create_method_data(chg_method, set_of_mols_par)
    create_method_data(chg_method, set_of_mols_val)
    if optimization_method == "optGM":
        create_method_data(chg_method, subset_of_mols)
        create_method_data(chg_method, min_subset_of_mols)
    print(colored("ok\n", "green"))

    print(f"Parameterizating by {' '.join(optimization_method.split('_'))}...")
    if optimization_method == "local_minimization":
        par_results = local_minimization(objective_function,
                                         set_of_mols_par,
                                         chg_method,
                                         chg_method.params_vals,
                                         maxiter)

    elif optimization_method == "optGM":
        par_results = optGM(objective_function,
                            set_of_mols_par,
                            subset_of_mols,
                            min_subset_of_mols,
                            chg_method,
                            num_of_samples,
                            num_of_candidates)

    elif optimization_method == "GDMIN":
        par_results = GDMIN(objective_function,
                            set_of_mols_par,
                            num_of_samples,
                            num_of_candidates,
                            chg_method)
    print(colored("\x1b[2Kok\n", "green"))

    chg_method.new_params(par_results.params,
                          set_of_mols.sdf_file,
                          f"{data_dir}/output_files/parameters.json",
                          params_file,
                          ref_chgs_file,
                          start_time.strftime('%Y-%m-%d %H:%M'))

    print("Preparing data for comparison...")
    set_of_mols_val.add_emp_chg(chg_method.calculate(set_of_mols_val),
                                chg_method.ats_types,
                                chg_method.params_per_at_type)
    set_of_mols_par.add_emp_chg(chg_method.calculate(set_of_mols_par),
                                chg_method.ats_types,
                                chg_method.params_per_at_type)
    results_full_set = chg_method.calculate(set_of_mols)
    print(colored("ok\n", "green"))

    write_chgs_to_file(results_full_set, set_of_mols)

    comparison_par(set_of_mols_par,
                   set_of_mols_val,
                   params_file,
                   data_dir,
                   par_results.loc_min_courses,
                   par_info=[f"Date of parameterization: {start_time.strftime('%Y-%m-%d %H:%M')}",
                             f"Time: {str(date.now() - start_time)[:-7]}\n\n",
                             f"Type of cpu: {[x.strip().split(':')[1] for x in open('/proc/cpuinfo').readlines() if 'model name' in x][0]}",
                             f"Command: {' '.join(argv)}",
                             f"Number of parameters: {len(chg_method.params_vals)}",
                             f"Objective function evaluations: {obj_eval}",
                             f"Objective function evaluations ended with error: {obj_eval_error}",
                             f"Achieved value of objective function: {par_results.obj_val}"])
