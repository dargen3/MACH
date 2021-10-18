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

    """
    ### tomas smazat
    # num_of_mols_par = set_of_mols_par.num_of_mols
    # num_of_mols_subset = min_subset_of_mols.num_of_mols
    #
    #
    #
    # samples = lhs(10000, chg_method.params_bounds)
    #
    # from time import time
    # s1 = time()
    # samples_par = [objective_function(sample, chg_method, set_of_mols_par) for sample in samples]
    # par_time = time() - s1
    #
    # s2 = time()
    # samples_subset = [objective_function(sample, chg_method, min_subset_of_mols) for sample in samples]
    # subset_time = time() - s2
    #
    # samples_ob = [[x,y] for x,y in zip(samples_par, samples_subset)]
    #
    # from operator import itemgetter
    # sorted_by_par = sorted(samples_ob, key=itemgetter(0))
    # for index, item in enumerate(sorted_by_par):
    #     item.append(index)
    #
    # sorted_by_subset = sorted(sorted_by_par, key=itemgetter(1))
    # for index, item in enumerate(sorted_by_subset):
    #     item.append(index)
    #
    # print("\n\n\n")
    # with open("samples_data_CCD_gen.txt", "w") as f:
    #     f.write(f"parameterization_set_mols: {num_of_mols_par}\n")
    #     f.write(f"subset_mols: {num_of_mols_subset}\n")
    #
    #     f.write(f"parameterization_set_time: {par_time} s\n")
    #     f.write(f"subset_time: {subset_time} s\n")
    #     f.write(f"300 parameterization_set_time {par_time*0.03} s\n")
    #
    #
    #     f.write("\n\nob_parameterization_set ob_subset rank_parameterization_set rank_subset")
    #     for item in sorted_by_subset:
    #         f.write(" ".join([str(x) for x in item]) + "\n")
    #
    #
    # from bokeh.plotting import show, figure
    #
    # graph = figure()
    # graph.circle([x[2] for x in sorted_by_subset], [x[3] for x in sorted_by_subset])
    # show(graph)
    #
    # exit()

    print("    Sampling...")
    samples = lhs(10, chg_method.params_bounds)

    print("    Calculating of objective function for samples...")
    samples_rmsd = [[sample, objective_function(sample, chg_method, set_of_mols_par)] for sample in samples]

    from operator import itemgetter
    best300 = sorted(samples_rmsd, key=itemgetter(1))[:5]

    from .GDMIN import local_minimization_NEWUOA
    import numpy as np

    def _rmsd_calculation(emp_chgs: np.array, set_of_mols) -> tuple:
        ats_types_rmsd = np.empty(len(chg_method.ats_types))
        for index, symbol in enumerate(chg_method.ats_types):
            differences = emp_chgs[set_of_mols.all_ats_ids == index * chg_method.params_per_at_type] - \
                          set_of_mols.ref_ats_types_chgs[symbol]
            ats_types_rmsd[index] = np.sqrt(np.mean(np.abs(differences) ** 2))
        total_mols_rmsd = 0
        index = 0
        for mol in set_of_mols.mols:
            new_index = index + mol.num_of_ats
            differences = emp_chgs[index: new_index] - mol.ref_chgs
            total_mols_rmsd += np.sqrt(np.mean(np.abs(differences) ** 2))
            index = new_index
        return  total_mols_rmsd / set_of_mols.num_of_mols


    with open("CCD_gen_300.txt", "a") as file:
        file.write(f"ID obj_before rmsd_before pearson_before obj_after rmsd_after pearson_after\n")
    for ID, sample in enumerate(best300):
        print(ID)
        chg_method.params_vals = sample[0]
        emp_chgs = chg_method.calculate(set_of_mols_par)

        obj_pred = sample[1]
        rmsd_pred = _rmsd_calculation(emp_chgs, set_of_mols_par)
        pearson_pred = np.corrcoef(set_of_mols_par.ref_chgs, emp_chgs)[0, 1]

        opt_par, obj_po,_,_ = local_minimization_NEWUOA(set_of_mols_par,
                                  chg_method,
                                  sample[0],
                                  10,
                                  objective_function)

        chg_method.params_vals = opt_par
        emp_chgs = chg_method.calculate(set_of_mols_par)

        rmsd_po = _rmsd_calculation(emp_chgs, set_of_mols_par)
        pearson_po = np.corrcoef(set_of_mols_par.ref_chgs, emp_chgs)[0, 1]
        with open("CCD_gen_300.txt", "a") as file:
            file.write(f"{ID} {obj_pred} {rmsd_pred} {pearson_pred} {obj_po} {rmsd_po} {pearson_po}\n")



    exit()
    ### tomas smazat
    """

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




def optGMp(objective_function: "function",
          set_of_mols_par: SetOfMolecules,
          subset_of_mols: SetOfMolecules,
          min_subset_of_mols: SetOfMolecules,
          chg_method: ChargeMethod,
          num_of_samples: int,
          num_of_candidates: int,
           bounds) -> namedtuple:

    print("    Sampling...")
    samples = lhs(num_of_samples, bounds)

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
