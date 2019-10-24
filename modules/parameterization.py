from .set_of_molecules import SetOfMolecules
from .control_existing import control_existing_files
from .output_files import write_charges_to_file
from .comparison import Comparison
from .molecule import MoleculeChg
from .optimization_methods import local_minimization, guided_minimization
from importlib import import_module
from termcolor import colored
from numba import jit
from sys import exit, argv
from itertools import chain
from datetime import datetime as date
from datetime import timedelta
from os import path, mkdir
from shutil import copyfile
from numpy import sum, sqrt, abs, max, linalg, array, mean, empty, isnan
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


def prepare_data_for_comparison(method, set_of_molecules, calculated_charges):
    atomic_types_charges = {}
    set_of_molecules.atomic_types_charges = {}
    for index, atomic_type_charges in enumerate(set_of_molecules.ref_atomic_types_charges):
        atomic_type = method.atomic_types[index]
        asn = set_of_molecules.all_symbolic_numbers_atoms
        atomic_types_charges[atomic_type] = calculated_charges[asn == index]
        set_of_molecules.atomic_types_charges[atomic_type] = set_of_molecules.ref_charges[asn == index]
    chg_molecules = []
    index = 0
    for molecule in set_of_molecules.molecules:
        molecule_len = len(molecule)
        chg_molecules.append(MoleculeChg(calculated_charges[index:index + molecule_len], molecule.name))
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


class Parameterization:
    def __init__(self, sdf, ref_charges, parameters, method, optimization_method, minimization_method, atomic_types_pattern, num_of_molecules, num_of_samples, num_of_candidates, subset_heuristic, validation, cpu, data_dir, rewriting_with_force, seed, git_hash=None):
        sdf_original = sdf
        start_time = date.now()
        files = [(sdf, True, "file"),
                 (ref_charges, True, "file"),
                 (data_dir, False, "directory")]
        if parameters is not None:
            files.extend([(parameters, True, "file")])
        control_existing_files(files,
                               rewriting_with_force)
        set_of_molecules = SetOfMolecules(sdf, num_of_molecules=num_of_molecules, parameterization=(ref_charges, validation), random_seed=seed)
        if len(sdf) < 2:
            exit(colored("Error! There must be more then 1 molecule for parameterization!\n"), "red")
        set_of_molecules_parameterization, set_of_molecules_validation = set_of_molecules.parameterization, set_of_molecules.validation
        method = getattr(import_module("modules.methods"), method)()
        method.load_parameters(parameters, set_of_molecules_parameterization, "parameterization", atomic_types_pattern=atomic_types_pattern)
        print("Preprocessing data...")
        set_of_molecules_parameterization.create_method_data(method)
        set_of_molecules_validation.create_method_data(method)
        print(colored("ok\n", "green"))

        print(f"Parameterizating by {' '.join(optimization_method.split('_'))}...")
        if optimization_method == "local_minimization":
            final_parameters = local_minimization(method.parameters_values, calculate_charges_and_statistical_data, minimization_method, method, set_of_molecules_parameterization)
        elif optimization_method == "guided_minimization":
            final_parameters = guided_minimization(calculate_charges_and_statistical_data, set_of_molecules_parameterization, method, num_of_samples, cpu, subset_heuristic, num_of_candidates, minimization_method)
        print(colored("ok\n", "green"))

        method.new_parameters(final_parameters)
        method.calculate(set_of_molecules_validation)
        results_validation = method.results



        method.calculate(set_of_molecules_parameterization)
        results_parameterization = method.results




        print("Preparing data for comparison...")
        atomic_types_charges_validation, chg_molecules_validation = prepare_data_for_comparison(method, set_of_molecules_validation, results_validation)
        atomic_types_charges, chg_molecules = prepare_data_for_comparison(method, set_of_molecules_parameterization, results_parameterization)
        print(colored("ok\n\n", "green") + "Copying files to {}...".format(data_dir))


        mkdir(data_dir)
        copyfile(sdf_original, path.join(data_dir, path.basename(sdf_original)))

        copyfile(ref_charges, path.join(data_dir, path.basename(ref_charges)))
        copyfile(parameters if parameters is not None else "modules/parameters/{}.json".format(str(method)), path.join(data_dir, "original_parameters.json"))
        print(colored("ok\n", "green"))


        set_of_molecules.create_method_data(method)
        method.calculate(set_of_molecules)
        charges = path.join(data_dir, path.basename(ref_charges)).replace(".chg", "_{}.chg".format(str(method)))
        write_charges_to_file(charges, method.results, set_of_molecules)
        summary_lines, parameters_json = write_parameters_to_file(path.join(data_dir, "parameters.json"), method, set_of_molecules.file, optimization_method, minimization_method, start_time, num_of_samples if optimization_method == "guided_minimization" else None, num_of_candidates if optimization_method == "guided_minimization" else None, cpu, git_hash, subset_heuristic if subset_heuristic != 0 else "False")




        comparison = Comparison(set_of_molecules_parameterization, (results_parameterization, atomic_types_charges, chg_molecules), data_dir, rewriting_with_force, parameterization=method, validation=[set_of_molecules_validation, (results_validation, atomic_types_charges_validation, chg_molecules_validation)])
        comparison.write_html_parameterization(path.join(data_dir, "{}_{}.html".format(path.basename(sdf_original)[:-4], method)), path.basename(sdf), path.basename(charges), path.basename(ref_charges), summary_lines, parameters_json)
