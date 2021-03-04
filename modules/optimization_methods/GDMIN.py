from collections import namedtuple
from heapq import nsmallest

import nlopt
import numpy as np

from .lhs import lhs
from ..chg_methods.chg_method import ChargeMethod
from ..set_of_molecules import SetOfMolecules


def GDMIN(objective_function: "function",
          set_of_mols: SetOfMolecules,
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
    candidates_rmsd = [objective_function(sample, chg_method, set_of_mols) for sample in samples]

    print("\x1b[2K    Selecting candidates...")
    main_candidates = samples[list(map(candidates_rmsd.index, nsmallest(num_of_candidates, candidates_rmsd)))]

    print(f"\x1b[2K    Local minimizating of best {num_of_candidates} candidates...")
    best_params = None
    best_obj_val = 1000000
    all_loc_min_course = []
    best_local_index = None
    for index, params in enumerate(main_candidates):
        opt_params, final_obj_val, loc_min_course, _ = local_minimization_NEWUOA(set_of_mols, chg_method, params,
                                                                                 num_of_samples, objective_function)
        all_loc_min_course.append(loc_min_course[0])
        if final_obj_val < best_obj_val:
            best_params = opt_params
            best_obj_val = final_obj_val
            best_local_index = index

    print(f"\x1b[2K    Local minimizating of best candidate...")
    best_opt_params, best_opt_val, best_opt_loc_min_course, _ = local_minimization_NEWUOA(set_of_mols, chg_method,
                                                                                          best_params,
                                                                                          3 * num_of_samples,
                                                                                          objective_function)
    all_loc_min_course[best_local_index].extend(best_opt_loc_min_course[0])
    return namedtuple("chgs", ["params",
                               "obj_val",
                               "loc_min_courses",
                               "obj_evals"])(best_opt_params,
                                             best_opt_val,
                                             all_loc_min_course,
                                             num_of_samples + sum(len(course) for course in all_loc_min_course))


def local_minimization_NEWUOA(set_of_mols: SetOfMolecules,
                              chg_method: ChargeMethod,
                              input_parameters: np.array,
                              steps: int,
                              objective_function: "function"):
    nlopt.srand(1)
    loc_min_course = []
    opt = nlopt.opt("LN_NEWUOA_BOUND", len(input_parameters))
    opt.set_min_objective(lambda x, grad: objective_function(x, chg_method, set_of_mols, loc_min_course))
    opt.set_xtol_abs(1e-6)
    opt.set_maxeval(steps)
    res = opt.optimize(input_parameters)
    return namedtuple("chgs", ["params",
                               "obj_val",
                               "loc_min_courses",
                               "obj_evals"])(res,
                                             opt.last_optimum_value(),
                                             [loc_min_course],
                                             len(loc_min_course))
