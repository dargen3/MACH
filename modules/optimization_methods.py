from heapq import nsmallest
from operator import itemgetter
from sys import stdout

from scipy.optimize import minimize

from .lhs import lhs


def local_minimization(input_parameters, objective_function, minimization_method, charge_method, set_of_molecules):
    if str(charge_method) in ["SFKEEM", "QEq", "MGC", "EQeq"]:
        bounds = [[0.000001, 100000] for _ in range(len(input_parameters))]
    else:
        bounds = [[-100000, 100000] for _ in range(len(input_parameters))]
    res = minimize(objective_function, input_parameters, method=minimization_method,
                   options={"maxiter": 10000}, bounds=bounds,
                   args=(charge_method, set_of_molecules))
    return res.x, res.fun


def modify_num_of_samples(num_of_samples, cpu):
    num_of_samples_cpu = num_of_samples / cpu
    iterations = int(num_of_samples_cpu / 20000) + 1
    chunksize = int(num_of_samples_cpu / iterations) + 1
    num_of_samples_modif = chunksize * cpu * iterations
    return num_of_samples_modif, chunksize


def guided_minimization(objective_function, set_of_molecules, charge_method, num_of_samples, cpu, num_of_candidates, minimization_method):
    print("    Sampling...")
    num_of_samples_modif, chunksize = modify_num_of_samples(num_of_samples, cpu)
    samples = lhs(len(charge_method.parameters_values), num_of_samples_modif, *charge_method.bounds)

    print("    Calculating of objective function for samples...")
    candidates_rmsd = [objective_function(sample, charge_method, set_of_molecules) for sample in samples]

    stdout.write('\x1b[2K')
    print("    Selecting candidates...")
    main_candidates = samples[list(map(candidates_rmsd.index, nsmallest(num_of_candidates, candidates_rmsd)))]

    print("    Local minimizating...")
    best_parameters = None
    best_rmsd = 1000000
    for parameters in main_candidates:
        opt_parameters, rmsd = local_minimization(parameters, objective_function, minimization_method, charge_method, set_of_molecules)
        if rmsd < best_rmsd:
            best_rmsd = rmsd
            best_parameters = opt_parameters


    return best_parameters