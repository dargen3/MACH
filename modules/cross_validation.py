from datetime import datetime as date
from importlib import import_module
from sys import argv
from os import makedirs

from termcolor import colored

from .comparison import comparison_par
from .control_input import control_and_copy_input_files
from .set_of_molecules import *
from .objective_function import ObjectiveFunction
from .optimization_methods.local_minimization import local_minimization
from .optimization_methods.optGM import optGM


def cross_validate(sdf_file: str,
                   ref_chgs_file: str,
                   chg_method: str,
                   params_file: str,
                   ats_types_pattern: str,
                   optimization_method: str,
                   num_of_samples: int,
                   num_of_candidates: int,
                   subset: int,
                   min_subset: int,
                   maxiter: int,
                   seed: int,
                   data_dir: str):

    control_and_copy_input_files(data_dir,
                                 (file for file in (sdf_file, ref_chgs_file, params_file) if file))

    for iter in range(5):
        iter_data_dir = data_dir + f"/output_files/{iter+1}"
        makedirs(iter_data_dir + "/output_files")

        start_time = date.now()
        chg_method = getattr(import_module(f"modules.chg_methods.{chg_method}"), chg_method)()
        ats_types_pattern = chg_method.load_params(params_file, ats_types_pattern)
        set_of_mols = create_set_of_mols(sdf_file, ats_types_pattern)
        set_of_mols.emp_chgs_file = f"{iter_data_dir}/output_files/empirical.chg"
        chg_method.prepare_params_for_par(set_of_mols)
        add_chgs(set_of_mols, ref_chgs_file, "ref_chgs")
        set_of_mols_par, set_of_mols_val = create_par_val_set(set_of_mols, seed, cross_validation_iter=iter)

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
        objective_function = ObjectiveFunction()
        if optimization_method == "local_minimization":
            par_results = local_minimization(objective_function.calculate,
                                             set_of_mols_par,
                                             chg_method,
                                             chg_method.params_vals,
                                             maxiter)

        elif optimization_method == "optGM":
            par_results = optGM(objective_function.calculate,
                                set_of_mols_par,
                                subset_of_mols,
                                min_subset_of_mols,
                                chg_method,
                                num_of_samples,
                                num_of_candidates)

        print(colored("\x1b[2Kok\n", "green"))

        chg_method.new_params(par_results.params,
                              set_of_mols.sdf_file,
                              f"{iter_data_dir}/output_files/parameters.json",
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
                       iter_data_dir,
                       par_results.loc_min_courses,
                       par_info=[f"Date of parameterization: {start_time.strftime('%Y-%m-%d %H:%M')}",
                                 f"Time: {str(date.now() - start_time)[:-7]}\n\n",
                                 f"Type of cpu: {[x.strip().split(':')[1] for x in open('/proc/cpuinfo').readlines() if 'model name' in x][0]}",
                                 f"Command: {' '.join(argv)}",
                                 f"Number of parameters: {len(chg_method.params_vals)}",
                                 f"Objective function evaluations: {objective_function.obj_eval}",
                                 f"Objective function evaluations ended with error: {objective_function.obj_eval_error}",
                                 f"Achieved value of objective function: {par_results.obj_val}"])
        chg_method = str(chg_method)
