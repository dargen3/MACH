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
def lhs(num_samples: int,
        bounds: np.array) -> np.array:

    dimensionality = len(bounds)
    cut = np.linspace(0, 1, num_samples + 1).astype(np.float32)
    samples = np.zeros((num_samples, dimensionality), dtype=np.float32)
    for x in range(num_samples):
        samples[x] += np.random.rand(dimensionality)
    for j in range(dimensionality):
        samples[np.random.permutation(np.arange(num_samples)), j] = samples[:, j] * cut[1] + cut[:num_samples]
    for x in range(num_samples):
        for y, (low_bound, high_bound) in enumerate(bounds):
            samples[x][y] = samples[x][y] * (high_bound - low_bound) + low_bound
    return samples


def local_minimization(set_of_mols_par: SetOfMolecules,
                       chg_method: ChargeMethod,
                       initial_params: np.array) -> namedtuple:

    loc_min_course = []
    res = minimize(objective_function,
                   initial_params,
                   method="SLSQP",
                   options={"maxiter": 1000000},
                   bounds=chg_method.params_bounds,
                   args=(chg_method, set_of_mols_par, loc_min_course))
    return namedtuple("chgs", ["params",
                               "obj_val",
                               "loc_min_courses",
                               "obj_evals"])(res.x,
                                             res.fun,
                                             [loc_min_course],
                                             len(loc_min_course))


def opt_guided_minimization(set_of_mols_par: SetOfMolecules,
                            set_of_mols_min: SetOfMolecules,
                            chg_method: ChargeMethod,
                            num_of_samples: int,
                            num_of_candidates: int) -> namedtuple:

    print("    Sampling...")
    samples = lhs(num_of_samples, chg_method.params_bounds)

    print("    Calculating of objective function for samples...")
    candidates_rmsd = [objective_function(sample, chg_method, set_of_mols_min) for sample in samples]

    print("\x1b[2K    Selecting candidates...")
    best_50_candidates = samples[list(map(candidates_rmsd.index, nsmallest(50, candidates_rmsd)))] # num_of_candidates
    best_50_candidates_rmsd = [objective_function(sample, chg_method, set_of_mols_par) for sample in best_50_candidates]
    best_5_candidates = best_50_candidates[list(map(best_50_candidates_rmsd.index, nsmallest(5, best_50_candidates_rmsd)))]  # num_of_candidates

    print("    Local minimizating...")
    best_params = None
    best_obj_val = 1000000
    all_loc_min_course = []
    for params in best_5_candidates:
        opt_params, final_obj_val, loc_min_course, _ = local_minimization(set_of_mols_par, chg_method, params)
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




def _local_minimization_nlopt(set_of_mols_80: SetOfMolecules,
                              chg_method: ChargeMethod,
                              input_parameters: np.array,
                              steps: int):
    import nlopt
    nlopt.srand(1)
    loc_min_course = []
    opt = nlopt.opt("LN_NEWUOA_BOUND", len(input_parameters))
    opt.set_min_objective(lambda x, grad: objective_function(x, chg_method, set_of_mols_80, loc_min_course))
    opt.set_xtol_abs(1e-6)
    opt.set_maxeval(steps)
    opt.set_lower_bounds([x[0] for x in chg_method.params_bounds])
    opt.set_upper_bounds([x[1] for x in chg_method.params_bounds])


    # print(opt.get_lower_bounds())
    # print(opt.get_upper_bounds())

    # try:
    res = opt.optimize(input_parameters)
    # except:
    #     print("\nERRROR\n")
    #     return namedtuple("chgs", ["params",
    #                            "obj_val",
    #                            "loc_min_courses",
    #                            "obj_evals"])(None,
    #                                          1000000000000000000000000000000,
    #                                          [],
    #                                          0)
    return namedtuple("chgs", ["params",
                               "obj_val",
                               "loc_min_courses",
                               "obj_evals"])(res,
                                             opt.last_optimum_value(),
                                             [loc_min_course],
                                             len(loc_min_course))


def guided_minimization(set_of_mols_80: SetOfMolecules,
                        num_of_samples: int,
                        num_of_candidates: int,
                        chg_method: ChargeMethod) -> namedtuple:

    # num_of_samples : 1000
    # num_of_candidates : 100
    # beg_iterations : 1000
    # end_interation : 3000

    print("    Sampling...")
    samples = lhs(num_of_samples, chg_method.params_bounds)

    print("    Calculating of objective function for samples...")
    candidates_rmsd = [objective_function(sample, chg_method, set_of_mols_80) for sample in samples]

    print("\x1b[2K    Selecting candidates...")
    main_candidates = samples[list(map(candidates_rmsd.index, nsmallest(num_of_candidates, candidates_rmsd)))]

    print(f"\x1b[2K    Local minimizating of best {num_of_candidates} candidates...")
    best_params = None
    best_obj_val = 1000000
    all_loc_min_course = []
    best_local_index = None
    for index, params in enumerate(main_candidates):
        opt_params, final_obj_val, loc_min_course, _ = _local_minimization_nlopt(set_of_mols_80, chg_method, params, num_of_samples)
        all_loc_min_course.append(loc_min_course[0])
        if final_obj_val < best_obj_val:
            best_params = opt_params
            best_obj_val = final_obj_val
            best_local_index = index

    print(f"\x1b[2K    Local minimizating of best candidate...")
    best_opt_params, best_opt_val, best_opt_loc_min_course, _ = _local_minimization_nlopt(set_of_mols_80, chg_method, best_params, 3*num_of_samples)
    print([len(x) for x in all_loc_min_course])
    all_loc_min_course[best_local_index].extend(best_opt_loc_min_course[0])
    print([len(x) for x in all_loc_min_course])

    return namedtuple("chgs", ["params",
                               "obj_val",
                               "loc_min_courses",
                               "obj_evals"])(best_opt_params,
                                             best_opt_val,
                                             all_loc_min_course,
                                             num_of_samples + sum(len(course) for course in all_loc_min_course))




def differential_evolution(set_of_mols_80: SetOfMolecules,
                           num_of_samples: int,
                           chg_method: ChargeMethod) -> namedtuple:

    print("    Sampling...")
    samples = lhs(num_of_samples, chg_method.params_bounds) # 1000



    print("    Calculating of objective function for samples...")
    candidates_rmsd = [objective_function(sample, chg_method, set_of_mols_80) for sample in samples]

    print("\x1b[2K    Selecting candidates...")
    candidates1 = samples[list(map(candidates_rmsd.index, nsmallest(100, candidates_rmsd)))] # 100

    print("    1. local minimization of candidates")
    loc_min_candidates1 = []
    best_val = 100000000
    best_candidate1 = None
    for candidate in candidates1:
        opt_params, obj_val,_,_ = _local_minimization_nlopt(set_of_mols_80, chg_method, candidate, 10000) # 1000
        loc_min_candidates1.append(opt_params)
        if obj_val < best_val:
            best_val = obj_val
            best_candidate1 = opt_params


    best_candidate2, _,_,_ = _local_minimization_nlopt(set_of_mols_80, chg_method, best_candidate1, 10000) # 1000

    ccc = 0
    ddd = 0
    from random import sample
    for x in range(num_of_samples):
        c1, c2 = sample(loc_min_candidates1, 2)
        trial = best_candidate2 + np.random.random() * (c1 - c2)

        error = False
        for bounds, parameter in zip(chg_method.params_bounds, trial):
            if bounds[0] > parameter or bounds[1] < parameter:
                error = True
        if error:
            continue



        obj_val = objective_function(trial, chg_method, set_of_mols_80)
        if obj_val < 1:
            ccc += 1
            opt_params, obj_val, _, _ = _local_minimization_nlopt(set_of_mols_80, chg_method, trial, 500)
            if obj_val < best_val:
                ddd += 1
                best_candidate2 = opt_params

    print(f"\n\n\n{ccc}  {ddd}\n\n\n")

    opt_params, obj_val, _, _ = _local_minimization_nlopt(set_of_mols_80, chg_method, best_candidate2, num_of_samples)

    return namedtuple("chgs", ["params",
                               "obj_val",
                               "loc_min_courses",
                               "obj_evals"])(best_candidate2,
                                             best_val,
                                             [],
                                             10)


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
        print("test\ntest\nest")
        return 1000

    ats_types_rmsd, rmsd = _rmsd_calculation(emp_chgs)
    objective_val = rmsd + np.mean(ats_types_rmsd)
    if np.isnan(objective_val):
        return 1000

    if isinstance(obj_vals, list):
        obj_vals.append(objective_val)
    print("    Total RMSD: {}    Worst RMSD: {}".format(str(rmsd)[:8], str(np.max(ats_types_rmsd))[:8]), end="\r")
    return objective_val


def parameterize(sdf_file: str,
                 ref_chgs_file: str,
                 chg_method: str,
                 params_file: str,
                 optimization_method: str,
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
    add_chgs(set_of_mols, ref_chgs_file, "ref_chgs")

    set_of_mols_80, set_of_mols_val = create_80_20(set_of_mols)
    if optimization_method in ["guided_minimization", "differential_evolution"]: # smazat
        print("Preprocessing data...")
        set_of_mols_par = set_of_mols_80
        create_method_data(chg_method, set_of_mols)
        create_method_data(chg_method, set_of_mols_par)
        create_method_data(chg_method, set_of_mols_val)
        print(colored("ok\n", "green"))

    else:
        set_of_mols_par, set_of_mols_min, _ = create_par_val_set(set_of_mols_80,
                                                                 subset,
                                                                 chg_method)
        print("Preprocessing data...")
        create_method_data(chg_method, set_of_mols)
        create_method_data(chg_method, set_of_mols_par)
        create_method_data(chg_method, set_of_mols_min)
        create_method_data(chg_method, set_of_mols_val)
        print(colored("ok\n", "green"))

    print(f"Parameterizating by {' '.join(optimization_method.split('_'))}...")
    if optimization_method == "local_minimization":
        par_results = local_minimization(set_of_mols_par,
                                         chg_method,
                                         chg_method.params_vals)
    elif optimization_method == "opt_guided_minimization":
        par_results = opt_guided_minimization(set_of_mols_par,
                                              set_of_mols_min,
                                              chg_method,
                                              num_of_samples,
                                              num_of_candidates)
    elif optimization_method == "guided_minimization":
        par_results = guided_minimization(set_of_mols_par,
                                          num_of_samples,
                                          num_of_candidates,
                                          chg_method)
    elif optimization_method == "differential_evolution":
        par_results = differential_evolution(set_of_mols_par,
                                             num_of_samples,
                                             chg_method)

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
