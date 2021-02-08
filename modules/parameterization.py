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

global obj_eval
global obj_eval_error
obj_eval = 0
obj_eval_error = 0

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

    # print("local minimization termination errrorororor!!") # dopsat do args num of ites
    # return namedtuple("chgs", ["params",
    #                       "obj_val",
    #                       "loc_min_courses"])(initial_params,
    #                                           0,
    #                                           [[0,0,0]])

    loc_min_course = []
    res = minimize(objective_function,
                   initial_params,
                   method="SLSQP",
                   options={"maxiter": 100000000000, "ftol": 1e-10},
                   args=(chg_method, set_of_mols_par, loc_min_course))
    return namedtuple("chgs", ["params",
                               "obj_val",
                               "loc_min_courses"])(res.x,
                                                   res.fun,
                                                   [loc_min_course])


def opt_guided_minimization(set_of_mols_par: SetOfMolecules,
                            subset_of_mols: SetOfMolecules,
                            min_subset_of_mols: SetOfMolecules,
                            chg_method: ChargeMethod,
                            num_of_samples: int,
                            num_of_candidates: int) -> namedtuple:

    print("    Sampling...")
    samples = lhs(num_of_samples, chg_method.params_bounds)

    print("    Calculating of objective function for samples...")
    samples_rmsd = [objective_function(sample, chg_method, min_subset_of_mols) for sample in samples]

    print("\x1b[2K    Selecting candidates...")
    best_samples = samples[list(map(samples_rmsd.index, nsmallest(num_of_candidates * 100, samples_rmsd)))]
    best_samples_rmsd = [objective_function(sample, chg_method, set_of_mols_par) for sample in best_samples]
    candidates = best_samples[list(map(best_samples_rmsd.index, nsmallest(num_of_candidates, best_samples_rmsd)))]

    print("\x1b[2K    Local minimizating...")
    all_loc_min_course = []
    opt_candidates = []
    for params in candidates:
        opt_params, _, loc_min_course = local_minimization(subset_of_mols, chg_method, params)
        all_loc_min_course.append(loc_min_course[0])
        opt_candidates.append(opt_params)

    opt_candidates_rmsd = [objective_function(candidate, chg_method, set_of_mols_par) for candidate in opt_candidates]
    final_candidate_obj_val = nsmallest(1, opt_candidates_rmsd)
    final_candidate_index = opt_candidates_rmsd.index(final_candidate_obj_val)
    final_candidate = opt_candidates[final_candidate_index]

    print("\x1b[2K    Final local minimizating...")
    final_params, final_obj_val, loc_min_course = local_minimization(set_of_mols_par, chg_method, final_candidate)
    all_loc_min_course[final_candidate_index].extend(loc_min_course[0])

    return namedtuple("chgs", ["params",
                               "obj_val",
                               "loc_min_courses"])(final_params,
                                                   final_obj_val,
                                                   all_loc_min_course)






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
    # opt.set_lower_bounds([x[0] for x in chg_method.params_bounds])
    # opt.set_upper_bounds([x[1] for x in chg_method.params_bounds])


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


    # from scipy.stats import pearsonr
    #
    # def pear():
    #     ats_types_pear = np.empty(len(chg_method.ats_types))
    #
    #     for index, symbol in enumerate(chg_method.ats_types):
    #         # print(symbol, pearsonr(emp_chgs[set_of_mols.all_ats_ids == index * chg_method.params_per_at_type], set_of_mols.ref_ats_types_chgs[symbol])[0])
    #
    #         ats_types_pear[index] = -pearsonr(emp_chgs[set_of_mols.all_ats_ids == index * chg_method.params_per_at_type], set_of_mols.ref_ats_types_chgs[symbol])[0]
    #     return ats_types_pear
    #
    # d = np.max(pear())
    # # print(d)
    # return  d + objective_val/2

    return objective_val





def objective_function_tomas(params: np.array,
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


    from scipy.stats import pearsonr

    def pear():
        ats_types_pear = np.empty(len(chg_method.ats_types))
        for index, symbol in enumerate(chg_method.ats_types):
            ats_types_pear[index] = pearsonr(emp_chgs[set_of_mols.all_ats_ids == index * chg_method.params_per_at_type], set_of_mols.ref_ats_types_chgs[symbol])[0]
        return ats_types_pear

    total_pearson = pearsonr(emp_chgs, set_of_mols.ref_chgs)[0]

    return rmsd, max(ats_types_rmsd), total_pearson, np.min(pear()),

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
                 seed: int,
                 data_dir: str,
                 rewriting_with_force: bool,
                 git_hash: str = None):


    # #1) celá sada molekul, ruzné samplování(5seedu), 1000000 samplů
    # chg_method = getattr(import_module("modules.chg_methods"), chg_method)()
    # ats_types_pattern = chg_method.load_params(params_file, ats_types_pattern)
    # set_of_mols = create_set_of_mols(sdf_file, ats_types_pattern)
    # set_of_mols.emp_chgs_file = f"{data_dir}/output_files/empirical.chg"
    # chg_method.prepare_params_for_par(set_of_mols)
    # add_chgs(set_of_mols, ref_chgs_file, "ref_chgs")
    # create_method_data(chg_method, set_of_mols)
    # print("    Sampling...")
    # samples = lhs(num_of_samples, chg_method.params_bounds)
    #
    # print("    Calculating of objective function for samples...")
    #
    # from time import time
    #
    # # start = time()
    # samples_rmsd = [objective_function_tomas(sample, chg_method, set_of_mols) for sample in samples]
    # # print(time()-start)
    # import csv
    # with open(f"CCD_gen_sampling_seed-{seed}.csv", 'w') as outcsv:
    #     # configure writer to write standard csv file
    #     writer = csv.writer(outcsv, delimiter=',')
    #     writer.writerow(['rmsd', 'worst at. type rmsd', 'pearson', "worst at. type pearson"])
    #     for item in samples_rmsd:
    #         # Write item to outcsv
    #         writer.writerow([item[0], item[1], item[2], item[3]])
    #
    # exit()
    # # 1 - end

    start_time = date.now()
    control_and_copy_input_files(data_dir,
                                 (file for file in (sdf_file, ref_chgs_file, params_file) if file),
                                 rewriting_with_force)

    chg_method = getattr(import_module("modules.chg_methods"), chg_method)()
    ats_types_pattern = chg_method.load_params(params_file, ats_types_pattern)
    set_of_mols = create_set_of_mols(sdf_file, ats_types_pattern)
    set_of_mols.emp_chgs_file = f"{data_dir}/output_files/empirical.chg"
    chg_method.prepare_params_for_par(set_of_mols)
    add_chgs(set_of_mols, ref_chgs_file, "ref_chgs")

    set_of_mols_par, set_of_mols_val = create_80_20(set_of_mols, percent_par, seed)
    subset_of_mols, min_subset_of_mols = create_par_val_set(set_of_mols_par,
                                                            subset)

    print("Preprocessing data...")
    create_method_data(chg_method, set_of_mols)
    create_method_data(chg_method, set_of_mols_par)
    create_method_data(chg_method, subset_of_mols)
    create_method_data(chg_method, min_subset_of_mols)
    create_method_data(chg_method, set_of_mols_val)
    print(colored("ok\n", "green"))

    print(f"Parameterizating by {' '.join(optimization_method.split('_'))}...")
    #if str(chg_method) == "SQEqp":
    #    for at_type in chg_method.ats_types:
    #        print(at_type, len(set_of_mols_par.ref_ats_types_chgs[at_type]), np.mean(set_of_mols_par.ref_ats_types_chgs[at_type]))
    #    exit()
    """d = []
    for mol in set_of_mols.mols:
        d.append(mol.total_chg)

    from collections import Counter
    c = Counter(d)
    print(c.most_common())
    exit()"""

    chg_method.average_charges = np.zeros(len(chg_method.ats_types)*3)
    for index, at_type in enumerate(chg_method.ats_types):

        chg_method.average_charges[index*3] = np.mean(set_of_mols_par.ref_ats_types_chgs[at_type])

    print([chg_method.average_charges[x*3] for x in range(len(chg_method.ats_types))])

    print(chg_method.ats_types)
    exit()
    if str(chg_method) == "SQEqa":
        def add_av_ref_charges(set_of_mols_a):
            average_charges_list = []
            for mol in set_of_mols_a.mols:
                average_charges_list.extend(molecules_average_charges[mol.name])
            set_of_mols_a.ref_av_charges = np.array(average_charges_list, dtype=np.float64)



        set_of_mols_plain_ba = create_set_of_mols(sdf_file, "plain-ba")
        add_chgs(set_of_mols_plain_ba, ref_chgs_file, "ref_chgs")
        average_charges = {}
        for at_type in set_of_mols_plain_ba.ats_types:
            average_charges[at_type] = np.mean(set_of_mols_plain_ba.ref_ats_types_chgs[at_type])

        molecules_average_charges = {}
        for molecule in set_of_mols_plain_ba.mols:
            data = []
            for atom in molecule.ats_srepr:
                data.append(average_charges[atom])
            molecules_average_charges[molecule.name] = data
        add_av_ref_charges(set_of_mols)
        add_av_ref_charges(set_of_mols_par)
        add_av_ref_charges(subset_of_mols)
        add_av_ref_charges(min_subset_of_mols)
        add_av_ref_charges(set_of_mols_val)


    if optimization_method == "local_minimization":
        par_results = local_minimization(set_of_mols_par,
                                         chg_method,
                                         chg_method.params_vals)
    elif optimization_method == "opt_guided_minimization":
        par_results = opt_guided_minimization(set_of_mols_par,
                                              subset_of_mols,
                                              min_subset_of_mols,
                                              chg_method,
                                              num_of_samples,
                                              num_of_candidates)
    elif optimization_method == "guided_minimization":
        par_results = guided_minimization(set_of_mols_par,
                                          num_of_samples,
                                          num_of_candidates,
                                          chg_method)
    elif optimization_method == "bayesian_optimization":
        from skopt import gp_minimize

        # res = minimize(objective_function,
        #                initial_params,
        #                method="SLSQP",
        #                options={"maxiter": 10000000, "ftol": 1e-10},
        #                args=(chg_method, set_of_mols_par, loc_min_course))
        # chg_method: ChargeMethod,
        # set_of_mols: SetOfMolecules,
        # obj_vals: list = None

        from functools import partial
        f = partial(objective_function, chg_method=chg_method, set_of_mols=subset_of_mols)
        res = gp_minimize(f, chg_method.params_bounds, n_calls=200, n_initial_points=100)



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
                             f"Achieved value of objective function: {par_results.obj_val}",
                             f"Github commit hash: <a href = \"https://github.com/dargen3/MACH/commit/{git_hash}\">{git_hash}</a></div>"])
