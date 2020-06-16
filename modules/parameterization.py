from collections import namedtuple
from datetime import datetime as date
from heapq import nsmallest
from importlib import import_module
from sys import argv

import numpy as np
from numba import jit
from scipy.optimize import minimize
from termcolor import colored

from .comparison import comparison_par
from .control_input import control_and_copy_input_files
from .set_of_molecules import *
from .chg_methods import ChargeMethod


@jit(nopython=True, cache=True)
def lhs(dimensionality: int,
        samples: int,
        high_bound: float,
        low_bound: float) -> np.array:

    cut = np.linspace(0, 1, samples + 1).astype(np.float32)
    u = np.zeros((samples, dimensionality), dtype=np.float32)
    for x in range(samples):
        u[x] += np.random.rand(dimensionality)
    for j in range(dimensionality):
        u[np.random.permutation(np.arange(samples)), j] = u[:, j] * cut[1] + cut[:samples]
    for x in range(samples):
        u[x] = u[x] * (high_bound - low_bound) + low_bound
    return u


def local_minimization(set_of_mols: SetOfMolecules,
                       chg_method: ChargeMethod,
                       minimization_method: str,
                       initial_params: np.array) -> namedtuple:

    loc_min_course = []
    res = minimize(objective_function,
                   initial_params,
                   method=minimization_method,
                   options={"maxiter": 1},
                   bounds=[chg_method.bounds for _ in range(len(initial_params))],
                   args=(chg_method, set_of_mols, loc_min_course))
    return namedtuple("chgs", ["params",
                                  "obj_val",
                                  "loc_min_courses",
                                  "obj_evals"])(res.x,
                                                res.fun,
                                                [loc_min_course],
                                                len(loc_min_course))


def guided_minimization(set_of_mols: SetOfMolecules,
                        chg_method: ChargeMethod,
                        num_of_samples: int,
                        num_of_candidates: int,
                        minimization_method: str) -> namedtuple:

    print("    Sampling...")
    samples = lhs(len(chg_method.params_vals), num_of_samples, *chg_method.bounds)

    print("    Calculating of objective function for samples...")
    candidates_rmsd = [objective_function(sample, chg_method, set_of_mols) for sample in samples]

    print("\x1b[2K    Selecting candidates...")
    main_candidates = samples[list(map(candidates_rmsd.index, nsmallest(num_of_candidates, candidates_rmsd)))]

    print("    Local minimizating...")
    best_params = None
    best_obj_val = 1000000
    all_loc_min_course = []
    for params in main_candidates:
        opt_params, final_obj_val, loc_min_course, _ = local_minimization(set_of_mols, chg_method, minimization_method, params)
        all_loc_min_course.append(loc_min_course[0])
        if final_obj_val < best_obj_val:
            best_params = opt_params
            best_obj_val = final_obj_val
    return namedtuple("chgs", ["params",
                                  "obj_val",
                                  "loc_min_courses",
                                  "obj_evals"])(best_params,
                                                best_obj_val,
                                                all_loc_min_course,
                                                num_of_samples + sum(len(course) for course in all_loc_min_course))


def objective_function(params: np.array,
                       chg_method: ChargeMethod,
                       set_of_mols: SetOfMolecules,
                       obj_vals: list = None) -> float:

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

    chg_method.params_vals = params
    try:
        emp_chgs = chg_method.calculate(set_of_mols)
    except (np.linalg.LinAlgError, ZeroDivisionError):
        return 1000

    ats_types_rmsd, rmsd = _rmsd_calculation(emp_chgs)
    print("    Total RMSD: {}    Worst RMSD: {}".format(str(rmsd)[:8], str(np.max(ats_types_rmsd))[:8]), end="\r")
    objective_val = rmsd + np.mean(ats_types_rmsd)
    if isinstance(obj_vals, list):
        obj_vals.append(objective_val)

    if np.isnan(objective_val):
        return 1000
    return objective_val


def parameterize(sdf_file: str,
                 ref_chgs_file: str,
                 chg_method: str,
                 params_file: str,
                 optimization_method: str,
                 minimization_method: str,
                 num_of_samples: int,
                 num_of_candidates: int,
                 subset: int,
                 data_dir: str,
                 rewriting_with_force: bool,
                 git_hash: str = None):

    start_time = date.now()
    control_and_copy_input_files(data_dir,
                                 (sdf_file, ref_chgs_file, params_file),
                                 rewriting_with_force)

    chg_method = getattr(import_module("modules.chg_methods"), chg_method)()
    ats_types_pattern = chg_method.load_params(params_file)
    set_of_mols = create_set_of_mols(sdf_file, ats_types_pattern)
    set_of_mols.emp_chgs_file = f"{data_dir}/output_files/empirical.chg"
    chg_method.prepare_params_for_par(set_of_mols)
    add_chgs_to_set_of_mols(set_of_mols, ref_chgs_file, "ref_chgs")
    set_of_mols_par, set_of_mols_val = create_par_val_set(set_of_mols,
                                                          subset,
                                                          chg_method)

    print("Preprocessing data...")
    create_method_data(chg_method, set_of_mols)
    create_method_data(chg_method, set_of_mols_par)
    create_method_data(chg_method, set_of_mols_val)
    print(colored("ok\n", "green"))

    print(f"Parameterizating by {' '.join(optimization_method.split('_'))}...")
    if optimization_method == "local_minimization":
        par_results = local_minimization(set_of_mols_par,
                                         chg_method,
                                         minimization_method,
                                         chg_method.params_vals)
    elif optimization_method == "guided_minimization":
        par_results = guided_minimization(set_of_mols_par,
                                          chg_method,
                                          num_of_samples,
                                          num_of_candidates,
                                          minimization_method)
    print(colored("\x1b[2Kok\n", "green"))

    chg_method.new_params(par_results.params,
                          set_of_mols.sdf_file,
                          f"{data_dir}/output_files/parameters.json",
                          ref_chgs_file,
                          params_file,
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
                             f"Objective function evaluations: {par_results.obj_evals}",
                             f"Github commit hash: "
                             f"<a href = \"https://github.com/dargen3/MACH/commit/{git_hash}\">{git_hash}</a></div>"])
