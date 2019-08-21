from .set_of_molecules import SubsetOfMolecules
from .lhs import lhs
from scipy.optimize import minimize
from heapq import nsmallest
from functools import partial
from multiprocessing import Pool
from operator import itemgetter


def local_minimization(input_parameters, objective_function, minimization_method, charge_method, set_of_molecules, submolecules):
    if str(charge_method) in ["SFKEEM", "QEq", "MGC"]:
        bounds = [[0.000001, 100000] for _ in range(len(input_parameters))]
    else:
        bounds = [[-100000, 100000] for _ in range(len(input_parameters))]

    res = minimize(objective_function, input_parameters, method=minimization_method,
                   options={"maxiter": 10000}, bounds=bounds,
                   args=(charge_method, set_of_molecules, submolecules))
    return res.fun, res.x


def modify_num_of_samples(num_of_samples, cpu):
    num_of_samples_cpu = num_of_samples / cpu
    iterations = int(num_of_samples_cpu / 20000) + 1
    chunksize = int(num_of_samples_cpu / iterations) + 1
    num_of_samples_modif = chunksize * cpu * iterations
    return num_of_samples_modif, chunksize


def guided_minimization(objective_function, set_of_molecules, charge_method, num_of_samples, cpu, subset_heuristic, submolecules, num_of_candidates, minimization_method):
    print("    Sampling...")
    num_of_samples_modif, chunksize = modify_num_of_samples(num_of_samples, cpu)
    samples = lhs(len(charge_method.parameters_values), num_of_samples_modif, *charge_method.bounds[0])

    print("    Calculating of objective function for samples...")
    partial_f = partial(objective_function, method=charge_method, set_of_molecules=set_of_molecules if subset_heuristic == 0 else SubsetOfMolecules(set_of_molecules, charge_method, subset_heuristic, submolecules), submolecules=submolecules)
    with Pool(cpu) as pool:
        candidates_rmsd = list(pool.imap(partial_f, samples, chunksize=chunksize))

    print("    Selecting candidates...")
    main_candidates = samples[list(map(candidates_rmsd.index, nsmallest(num_of_candidates, candidates_rmsd)))]

    print("    Local minimizating...")
    if subset_heuristic and str(charge_method) in ["EEM", "SFKEEM", "ACKS2", "QEq", "Comba"]:
        partial_f = partial(local_minimization, objective_function=objective_function, minimization_method=minimization_method, charge_method=charge_method, set_of_molecules=SubsetOfMolecules(set_of_molecules, charge_method, subset_heuristic * 3, submolecules), submolecules=submolecules)
        with Pool(cpu) as pool:
            main_candidates = [result[1] for result in pool.map(partial_f, [parameters for parameters in main_candidates])]
    partial_f = partial(local_minimization, objective_function=objective_function, minimization_method=minimization_method, charge_method=charge_method, set_of_molecules=set_of_molecules, submolecules=submolecules)
    with Pool(cpu) as pool:
        best_candidates = [result for result in pool.map(partial_f, [parameters for parameters in main_candidates])]
    best_candidates.sort(key=itemgetter(0))
    return best_candidates[0][1]
