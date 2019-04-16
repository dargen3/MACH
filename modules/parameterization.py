from .set_of_molecules import SetOfMolecules, SubsetOfMolecules
from .control_existing import control_existing_files
from .calculation import write_charges_to_file
from .comparison import Comparison
from .molecule import MoleculeChg
from importlib import import_module
from termcolor import colored
from scipy.optimize import minimize, differential_evolution
from numba import jit
from sys import exit, argv
from functools import partial
from multiprocessing import Pool
from itertools import chain
from operator import itemgetter
from datetime import datetime as date
from datetime import timedelta
from os import path, mkdir
from shutil import copyfile
import nlopt
from numpy import sum, sqrt, abs, max, array_split, linalg, array, linspace, random, arange, mean, empty, isnan, float32, zeros
import heapq
import git
from json import dumps


def local_minimization(input_parameters, minimization_method, method, set_of_molecules):
    if str(method) in ["SFKEEM", "QEq", "MGC"]:
        bounds = [[0.000001, 100000] for _ in range(len(input_parameters))]
    else:
        bounds = [[-100000, 100000] for _ in range(len(input_parameters))]

    res = minimize(calculate_charges_and_statistical_data, input_parameters, method=minimization_method, options={"maxiter": 10000}, bounds=bounds,
                   args=(method, set_of_molecules))
    return res.fun, res.x


def modify_num_of_samples(num_of_samples, cpu):
    num_of_samples_cpu = num_of_samples / cpu
    iterations = int(num_of_samples_cpu / 20000) + 1
    chunksize = int(num_of_samples_cpu / iterations) + 1
    num_of_samples_modif = chunksize * cpu * iterations
    return num_of_samples_modif, chunksize


def write_parameters_to_file(parameters_file, method, set_of_molecules_file, optimization_method, minimization_method, start_time, num_of_samples, num_of_candidates, cpu, git_hash, subset_heuristic):
    print("Writing parameters to {}...".format(parameters_file))
    if not git_hash:
        git_hash = git.Repo(search_parent_directories=True).head.object.hexsha
    summary_lines = ["Set of molecules: {}".format(set_of_molecules_file),
                     "Method: {}".format(method),
                     "Optimization method: {}".format(optimization_method),
                     "Minimization method: {}".format(minimization_method),
                     "Date of parameterization: {}".format(start_time.strftime("%Y-%m-%d %H:%M")),
                     "Time: {}\n\n".format(str(date.now() - start_time)[:-7]),
                     "Number of cpu: {}".format(cpu),
                     "Command: {}".format(" ".join(argv)),
                     "Github commit hash: <a href = \"{}\">{}</a></div>".format("https://github.com/dargen3/MACH/commit/{}".format(git_hash), git_hash)]
    if optimization_method == "guided_minimization":
        summary_lines.insert(3, "Number of samples: {}".format(num_of_samples))
        summary_lines.insert(3, "Number of candidates: {}".format(num_of_candidates))
        summary_lines.insert(3, "Subset heuristic: {}".format(subset_heuristic))
    parameters_json = dumps(method.parameters_json, indent=2, sort_keys=True)
    with open(parameters_file, "w") as par_file:
        par_file.write(parameters_json)
    print(colored("ok\n", "green"))
    return summary_lines, parameters_json


def prepare_data_for_comparison(method, set_of_molecules):
    atomic_types_charges = {}
    set_of_molecules.atomic_types_charges = {}
    for index, atomic_type_charges in enumerate(set_of_molecules.ref_atomic_types_charges):
        atomic_type, asn = method.atomic_types[index], set_of_molecules.all_symbolic_numbers_atoms
        atomic_types_charges[atomic_type] = method.results[asn == index]
        set_of_molecules.atomic_types_charges[atomic_type] = set_of_molecules.ref_charges[asn == index]
    chg_molecules = []
    index = 0
    for molecule in set_of_molecules.molecules:
        molecule_len = len(molecule)
        chg_molecules.append(MoleculeChg(method.results[index:index + molecule_len], molecule.name))
        index += molecule_len
    return atomic_types_charges, chg_molecules


@jit(nopython=True, cache=True)
def rmsd_calculation_atomic_type(charges, symbolic_numbers_atoms, atomic_type_symbolic_number, right_charges):
    deviations = abs(charges[symbolic_numbers_atoms == atomic_type_symbolic_number] - right_charges) ** 2
    return sqrt(mean(deviations))


@jit(nopython=True)
def rmsd_calculation_molecules(all_num_of_atoms, results, ref_charges):
    total_molecules_rmsd = 0
    index = 0
    for num_of_atoms in all_num_of_atoms:
        new_index = index + num_of_atoms
        total_molecules_rmsd += sqrt(mean(abs(results[index: new_index] - ref_charges[index: new_index]) ** 2))
        index = new_index
    return total_molecules_rmsd / all_num_of_atoms.size

def calculate_charges_and_statistical_data(list_of_parameters, method, set_of_molecules):
    method.parameters_values = list_of_parameters
    try:
        method.calculate(set_of_molecules)
    except (linalg.linalg.LinAlgError, ZeroDivisionError):
        return 1000
    results = method.results
    rmsd = rmsd_calculation_molecules(set_of_molecules.all_num_of_atoms, results, set_of_molecules.ref_charges)
    atomic_types_rmsd = [rmsd_calculation_atomic_type(results, set_of_molecules.all_symbolic_numbers_atoms, index, atomic_type_charges)
                         for index, atomic_type_charges in enumerate(set_of_molecules.ref_atomic_types_charges)]
    greater_rmsd = max(atomic_types_rmsd)
    print("Total RMSD: {}    Worst RMSD: {}".format(str(rmsd)[:8], str(greater_rmsd)[:8]), end="\r")
    objective_value = rmsd + mean(atomic_types_rmsd)
    if isnan(objective_value):
        return 1000
    return objective_value


@jit(nopython=True, cache=True)
def lhsclassic(n, samples, high_bound, low_bound):
    cut = linspace(0, 1, samples + 1).astype(float32)
    u = zeros((samples, n), dtype=float32)
    for x in range(samples):
        u[x] += random.rand(n)
    for j in range(n):
        u[random.permutation(arange(samples)), j] = u[:, j] * cut[1] + cut[:samples]
    for x in range(samples):
        u[x] = u[x] * (high_bound - low_bound) + low_bound
    return u


class Parameterization:
    def __init__(self, sdf, ref_charges, parameters, method, optimization_method, minimization_method, atomic_types_pattern, num_of_molecules, num_of_samples, num_of_candidates, subset_heuristic, validation, cpu, data_dir, rewriting_with_force, git_hash=None):
        start_time = date.now()
        files = [(sdf, True, "file"),
                 (ref_charges, True, "file"),
                 (data_dir, False, "directory")]
        if parameters is not None:
            files.extend([(parameters, True, "file")])
        control_existing_files(files,
                               rewriting_with_force)
        if num_of_molecules is None:
            num_of_molecules = open(sdf, "r").read().count("$$$$")
        if num_of_molecules < 2:
            pass
            exit(colored("ERROR! There must be more then 1 molecule for parameterization!\n", "red"))
        num_of_molecules_validation = int((1 - validation / 100) * num_of_molecules)
        set_of_molecules = SetOfMolecules(sdf, num_of_molecules_to=num_of_molecules_validation)
        method = getattr(import_module("modules.methods"), method)()
        method.load_parameters(parameters, set_of_molecules, "parameterization", atomic_types_pattern=atomic_types_pattern)
        set_of_molecules.create_method_data(method)
        set_of_molecules.add_ref_charges(ref_charges, len(method.atomic_types))
        set_of_molecules_validation = SetOfMolecules(sdf, num_of_molecules_from=num_of_molecules_validation, num_of_molecules_to=num_of_molecules)
        try:
            set_of_molecules_validation.create_method_data(method)
        except ValueError:
            exit(colored("ERROR! There are atomic types in last {}% of set of molecules, that are not in first {}%!\n".format(validation, 100 - validation), "red"))
        set_of_molecules_validation.add_ref_charges(ref_charges, len(method.atomic_types))
        print("Parameterizating...")
        if optimization_method == "local_minimization":
            _, final_parameters = local_minimization(method.parameters_values, minimization_method, method, set_of_molecules)
        elif optimization_method == "guided_minimization":
            num_of_samples_modif, chunksize = modify_num_of_samples(num_of_samples, cpu)
            samples = lhsclassic(len(method.parameters_values), num_of_samples_modif, *method.bounds[0])
            partial_f = partial(calculate_charges_and_statistical_data, method=method, set_of_molecules=set_of_molecules if subset_heuristic == 0 else SubsetOfMolecules(set_of_molecules, method, subset_heuristic))
            with Pool(cpu) as pool:
                candidates_rmsd = list(pool.imap(partial_f, samples, chunksize=chunksize))
            main_candidates = samples[list(map(candidates_rmsd.index, heapq.nsmallest(num_of_candidates, candidates_rmsd)))]
            if subset_heuristic and method in ["EEM", "SFKEEM", "ACKS2", "QEq"]:
                partial_f = partial(local_minimization, minimization_method=minimization_method, method=method, set_of_molecules=SubsetOfMolecules(set_of_molecules, method, subset_heuristic * 3))
                with Pool(cpu) as pool:
                    main_candidates = [result[1] for result in pool.map(partial_f, [parameters for parameters in main_candidates])]
            partial_f = partial(local_minimization, minimization_method=minimization_method, method=method, set_of_molecules=set_of_molecules)
            with Pool(cpu) as pool:
                best_candidates = [result for result in pool.map(partial_f, [parameters for parameters in main_candidates])]
            best_candidates.sort(key=itemgetter(0))
            final_parameters = best_candidates[0][1]
        method.new_parameters(final_parameters)
        method.calculate(set_of_molecules_validation)
        results_validation = method.results
        atomic_types_charges_validation, chg_molecules_validation = prepare_data_for_comparison(method, set_of_molecules_validation)
        method.calculate(set_of_molecules)
        print(colored("\033[Kok\n", "green"))
        atomic_types_charges, chg_molecules = prepare_data_for_comparison(method, set_of_molecules)
        mkdir(data_dir)
        comparison = Comparison(set_of_molecules, (method.results, atomic_types_charges, chg_molecules), data_dir, rewriting_with_force, parameterization=method, validation=[set_of_molecules_validation, (results_validation, atomic_types_charges_validation, chg_molecules_validation)])
        copyfile(sdf, path.join(data_dir, path.basename(sdf)))
        copyfile(ref_charges, path.join(data_dir, path.basename(ref_charges)))
        copyfile(parameters if parameters is not None else "modules/parameters/{}.json".format(str(method)), path.join(data_dir, "original_parameters.json"))
        charges = path.join(data_dir, path.basename(ref_charges)).replace(".chg", "_{}.chg".format(str(method)))
        write_charges_to_file(charges, method.results, set_of_molecules)
        summary_lines, parameters_json = write_parameters_to_file(path.join(data_dir, "parameters.json"), method, set_of_molecules.file, optimization_method, minimization_method, start_time, num_of_samples if optimization_method == "guided_minimization" else None, num_of_candidates if optimization_method == "guided_minimization" else None, cpu, git_hash, subset_heuristic if subset_heuristic != 0 else "False")
        comparison.write_html_parameterization(path.join(data_dir, "{}_{}.html".format(path.basename(sdf)[:-4], method)), path.basename(sdf), path.basename(charges), path.basename(ref_charges), summary_lines, parameters_json)
