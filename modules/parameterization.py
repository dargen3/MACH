from .set_of_molecules import create_set_of_molecules, create_method_data, create_parameterization_validation_set
from .control_existing import control_existing_files
from .input_output import write_charges_to_file, add_charges_to_set_of_molecules
from .comparison import Comparison
from .optimization_methods import local_minimization, guided_minimization
from importlib import import_module
from termcolor import colored
from numba import jit
from numba.types import float32, int16, string
from numba.typed import Dict
from sys import exit, argv
from itertools import chain
from datetime import datetime as date
from datetime import timedelta
from os import path, mkdir
from shutil import copyfile
from numpy import sum, sqrt, abs, max, linalg, array, mean, empty, isnan, float32 as npfloat32
import git
from json import dumps


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
                     "Type of cpu: {}".format([x.strip().split(":")[1] for x in open("/proc/cpuinfo").readlines() if "model name" in x][0]),
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


def prepare_data_for_comparison(set_of_molecules, emp_charges):
    set_of_molecules.emp_charges = emp_charges.astype(npfloat32)

    set_of_molecules.emp_atomic_types_charges = Dict.empty(key_type=string, value_type=float32[:])
    for index, symbol in enumerate(set_of_molecules.atomic_types):
        emp_atomic_type_charges = emp_charges[set_of_molecules.all_atoms_id == index*set_of_molecules.parameters_per_atomic_type].astype(npfloat32)
        if len(emp_atomic_type_charges):
            set_of_molecules.emp_atomic_types_charges[symbol] = emp_atomic_type_charges

    index = 0
    for molecule in set_of_molecules.molecules:
        molecule.emp_charges = emp_charges[index: index + molecule.num_of_atoms].astype(npfloat32)
        index += molecule.num_of_atoms


def calculate_charges_and_statistical_data(list_of_parameters, method, set_of_molecules):
    @jit(nopython=True, cache=True)
    def rmsd_calculation(set_of_molecules, emp_charges):
        atomic_types_rmsd = empty(len(set_of_molecules.atomic_types))
        for index, symbol in enumerate(set_of_molecules.atomic_types):
            atomic_types_rmsd[index] = sqrt(mean(abs(emp_charges[set_of_molecules.all_atoms_id == index * set_of_molecules.parameters_per_atomic_type] - set_of_molecules.ref_atomic_types_charges[symbol]) ** 2))
        total_molecules_rmsd = 0
        index = 0
        for molecule in set_of_molecules.molecules:
            new_index = index + molecule.num_of_atoms
            total_molecules_rmsd += sqrt(mean(abs(emp_charges[index: new_index] - molecule.ref_charges) ** 2))
            index = new_index
        return atomic_types_rmsd, total_molecules_rmsd / set_of_molecules.num_of_molecules

    method.parameters_values = list_of_parameters
    try:
        method.calculate(set_of_molecules)
    except (linalg.linalg.LinAlgError, ZeroDivisionError):
        return 1000
    results = method.results
    atomic_types_rmsd, rmsd = rmsd_calculation(set_of_molecules, results)
    print("Total RMSD: {}    Worst RMSD: {}".format(str(rmsd)[:8], str(max(atomic_types_rmsd))[:8]), end="\r")
    objective_value = rmsd + mean(atomic_types_rmsd)
    if isnan(objective_value):
        return 1000
    return objective_value


class Parameterization:
    def __init__(self, sdf_file, ref_chg_file, parameters, method, optimization_method, minimization_method, atomic_types_pattern, num_of_molecules, num_of_samples, num_of_candidates, subset_heuristic, validation, cpu, data_dir, rewriting_with_force, seed, git_hash=None):
        start_time = date.now()
        files = [(sdf_file, True, "file"),
                 (ref_chg_file, True, "file"),
                 (data_dir, False, "directory")]
        if parameters is not None:
            files.extend([(parameters, True, "file")])
        control_existing_files(files,
                               rewriting_with_force)

        set_of_molecules = create_set_of_molecules(sdf_file, atomic_types_pattern, num_of_molecules=num_of_molecules)
        add_charges_to_set_of_molecules(set_of_molecules, ref_chg_file)
        set_of_molecules_parameterization, set_of_molecules_validation = create_parameterization_validation_set(set_of_molecules, seed, validation, method)
        method = getattr(import_module("modules.methods"), method)()
        method.load_parameters(parameters, set_of_molecules_parameterization, "parameterization", atomic_types_pattern=atomic_types_pattern)

        print("Preprocessing data...")
        create_method_data(method, set_of_molecules_parameterization)
        create_method_data(method, set_of_molecules_validation)
        print(colored("ok\n", "green"))

        print(f"Parameterizating by {' '.join(optimization_method.split('_'))}...")
        if optimization_method == "local_minimization":
            final_parameters = local_minimization(method.parameters_values, calculate_charges_and_statistical_data, minimization_method, method, set_of_molecules_parameterization)
        elif optimization_method == "guided_minimization":
            final_parameters = guided_minimization(calculate_charges_and_statistical_data, set_of_molecules_parameterization, method, num_of_samples, cpu, subset_heuristic, num_of_candidates, minimization_method)
        method.new_parameters(final_parameters)
        method.calculate(set_of_molecules_validation)
        results_validation = method.results
        method.calculate(set_of_molecules_parameterization)
        results_parameterization = method.results
        create_method_data(method, set_of_molecules)
        method.calculate(set_of_molecules)
        results_full_set = method.results
        print(colored("ok\n", "green"))

        print("Preparing data for comparison...")
        prepare_data_for_comparison(set_of_molecules_validation, results_validation)
        prepare_data_for_comparison(set_of_molecules_parameterization, results_parameterization)
        print(colored("ok\n\n", "green") + "Copying files to {}...".format(data_dir))
        mkdir(data_dir)
        copyfile(sdf_file, path.join(data_dir, path.basename(sdf_file)))
        copyfile(ref_chg_file, path.join(data_dir, path.basename(ref_chg_file)))
        copyfile(parameters if parameters is not None else "modules/parameters/{}.json".format(str(method)), path.join(data_dir, "original_parameters.json"))
        print(colored("ok\n", "green"))

        emp_chg_file = path.join(data_dir, path.basename(ref_chg_file)).replace(".chg", "_{}.chg".format(str(method)))
        write_charges_to_file(emp_chg_file, results_full_set, set_of_molecules)
        summary_lines, parameters_json = write_parameters_to_file(path.join(data_dir, "parameters.json"), method, set_of_molecules.sdf_file, optimization_method, minimization_method, start_time, num_of_samples if optimization_method == "guided_minimization" else None, num_of_candidates if optimization_method == "guided_minimization" else None, cpu, git_hash, subset_heuristic if subset_heuristic != 0 else "False")

        Comparison().parameterization(set_of_molecules_parameterization,
                                      set_of_molecules_validation,
                                      path.join(data_dir, "{}_{}.html".format(path.basename(sdf_file)[:-4], method)),
                                      path.basename(sdf_file),
                                      path.basename(emp_chg_file),
                                      path.basename(ref_chg_file),
                                      summary_lines,
                                      parameters_json)
