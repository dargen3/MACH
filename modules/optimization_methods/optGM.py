from collections import namedtuple
from heapq import nsmallest

from .lhs import lhs
from .local_minimization import local_minimization
from ..chg_methods.chg_method import ChargeMethod
from ..set_of_molecules import SetOfMolecules


def optGM(objective_function: "function",
          set_of_mols_par: SetOfMolecules,
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
        opt_params, _, loc_min_course = local_minimization(objective_function, subset_of_mols, chg_method, params)
        all_loc_min_course.append(loc_min_course[0])
        opt_candidates.append(opt_params)

    opt_candidates_rmsd = [objective_function(candidate, chg_method, set_of_mols_par) for candidate in opt_candidates]
    final_candidate_obj_val = nsmallest(1, opt_candidates_rmsd)
    final_candidate_index = opt_candidates_rmsd.index(final_candidate_obj_val)
    final_candidate = opt_candidates[final_candidate_index]

    print("\x1b[2K    Final local minimizating...")
    final_params, final_obj_val, loc_min_course = local_minimization(objective_function, set_of_mols_par, chg_method, final_candidate)
    all_loc_min_course[final_candidate_index].extend(loc_min_course[0])

    return namedtuple("chgs", ["params",
                               "obj_val",
                               "loc_min_courses"])(final_params,
                                                   final_obj_val,
                                                   all_loc_min_course)
