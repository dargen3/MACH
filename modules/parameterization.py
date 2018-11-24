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
from multiprocessing import Pool, Manager
from itertools import chain
from operator import itemgetter
from datetime import datetime as date
from datetime import timedelta
from os import path, mkdir
from shutil import copyfile
import nlopt
from numpy import sum, sqrt, abs, max, array_split, linalg, array, linspace, random, arange
import heapq
import git

def local_minimization(input_parameters, minimization_method, method, set_of_molecules, bounds):
    course = []
    if minimization_method in ["SLSQP"]:
        res = minimize(calculate_charges_and_statistical_data, input_parameters, method=minimization_method, bounds=bounds,
                       args=(method, set_of_molecules, course), options={"disp": True})
        return res.fun, res.x, course
    elif minimization_method in ["NEWUOA"]:
        opt = nlopt.opt(nlopt.LN_NEWUOA, len(input_parameters))
        opt.set_min_objective(lambda x, grad: calculate_charges_and_statistical_data(x, method, set_of_molecules, course))
        opt.set_lower_bounds([x[0] for x in bounds])
        opt.set_upper_bounds([x[1] for x in bounds])
        opt.set_xtol_rel(1e-6)
        res = opt.optimize(input_parameters)
        print("\nNumber of steps: {}\n".format(opt.get_numevals()))
        return opt.last_opt_value(), res, course


def write_parameters_to_file(parameters_file, method, set_of_molecules_file, optimization_method, minimization_method, start_time, num_of_samples, cpu):
    print("Writing parameters to {}...".format(parameters_file))
    values = []
    for atomic_type in method.atomic_types:
        line = "{}  {}".format(atomic_type.replace("~", " "), "  ".join([str(round(method.parameters["{}_{}".format(atomic_type, key)], 4)) for key in method.atom_value_symbols]))
        for bonded_atom in method.bond_value_symbols:
            key = "bond-{}{}".format(atomic_type, bonded_atom)
            line += "  {}".format(round(method.parameters[key], 4)) if key in method.parameters else "   X"
        values.append(line)
    parameters_lines = ["method: {}".format(method),
                        "length_type: {}".format(method.length_correction_key),
                        "<<global>>"] + \
                       ["{}: {}".format(key, round(value, 4)) for key, value in method.parameters.items() if key[0].islower() and key[:5] != "bond-"] + \
                       ["<<key>>"] + \
                       [key for key in method.keys] + \
                       ["<<value_symbol>>"] + \
                       [key for key in method.value_symbols] + \
                       ["<<value>>"] + values  + ["<<end>>"]
    git_hash = git.Repo(search_parent_directories=True).head.object.hexsha
    summary_lines = ["Set of molecules: {}".format(set_of_molecules_file),
                     "Method: {}".format(method),
                     "Optimization method: {}".format(optimization_method),
                     "Date of parameterization: {}".format(start_time.strftime("%Y-%m-%d %H:%M")),
                     "CPU time: {}\n\n".format(str((date.now() - start_time)*cpu)[:-7]),
                     "Number of cpu: {}".format(cpu),
                     "Command: {}".format(" ".join(argv)),
                     "Github commit hash: <a href = \"{}\">{}</a></div>".format("https://github.com/dargen3/MACH/commit/{}".format(git_hash), git_hash)]
    if optimization_method in ["minimization", "guided_minimization"]:
        summary_lines.insert(3, "Minimization method: {}".format(minimization_method))
    if optimization_method == "guided_minimization":
        summary_lines.insert(3, "Samples: {}".format(num_of_samples))
    with open(parameters_file, "w") as par_file:
        par_file.writelines("\n".join(parameters_lines) + "\n\n\n")
        par_file.writelines("\n".join(summary_lines))
    print(colored("ok\n", "green"))
    return summary_lines, parameters_lines


def prepare_data_for_comparison(method, set_of_molecules):
    atomic_types_charges = {}
    set_of_molecules.atomic_types_charges = {}
    for index, atomic_type_charges in enumerate(set_of_molecules.ref_atomic_types_charges):
        atomic_type, asn = method.atomic_types[index], set_of_molecules.all_symbolic_numbers_atoms
        atomic_types_charges[atomic_type] = separate_atomic_type_charges(method.results, asn, index)
        set_of_molecules.atomic_types_charges[atomic_type] = separate_atomic_type_charges(set_of_molecules.ref_charges, asn, index)
    chg_molecules = []
    index = 0
    for molecule in set_of_molecules.molecules:
        molecule_len = len(molecule)
        chg_molecules.append(MoleculeChg(method.results[index:index + molecule_len], molecule.name))
        index += molecule_len
    return atomic_types_charges, chg_molecules


def prepare_data_for_parameterization(set_of_molecules, ref_charges, method):
    set_of_molecules.create_method_data(method)
    set_of_molecules.add_ref_charges(ref_charges, len(method.atomic_types))
    method.control_parameters(set_of_molecules.file, set_of_molecules.all_symbolic_numbers_atoms)

@jit(nopython=True, cache=True)
def rmsd_calculation(results, right_charges):
    deviations = abs(results - right_charges)**2
    return sqrt(sum(deviations) / deviations.size)


@jit(nopython=True, cache=True)
def separate_atomic_type_charges(charges, symbolic_numbers_atoms, atomic_type_symbolic_number):
    return charges[symbolic_numbers_atoms == atomic_type_symbolic_number]


def calculate_charges_and_statistical_data(list_of_parameters, method, set_of_molecules, course=None):
    method.parameters_values = list_of_parameters
    try:
        method.calculate(set_of_molecules)
    except (linalg.linalg.LinAlgError, ZeroDivisionError):
        return 1000
        # přepsat
    # rmsd pro jednotlivé molekuly, když bude rmsd vetší než...
    rmsd = rmsd_calculation(method.results, set_of_molecules.ref_charges)
    atomic_types_rmsd = []
    for index, atomic_type_charges in enumerate(set_of_molecules.ref_atomic_types_charges):
        calculated_atomic_types_charges = separate_atomic_type_charges(method.results, set_of_molecules.all_symbolic_numbers_atoms, index)
        atomic_types_rmsd.append(rmsd_calculation(calculated_atomic_types_charges, atomic_type_charges))
    greater_rmsd = max(atomic_types_rmsd)
    print("Total RMSD: {}    Worst RMSD: {}".format(str(rmsd)[:8], str(greater_rmsd)[:8]), end="\r")
    objective_value = greater_rmsd + rmsd + sum(atomic_types_rmsd) / len(atomic_types_rmsd)
    if course is not None:
        course.append(rmsd)
    """
    if objective_value > 500:
        print(objective_value)

        greater_than_objf = 0
        lower_than_objf = 0
        
        for index, x in enumerate(abs(method.results - set_of_molecules.ref_charges)):
            if x > objective_value:
                greater_than_objf += 1
                print(index, x)
            else:
                lower_than_objf += 1
        
        print(lower_than_objf, greater_than_objf)
        print()
        print()
        print()
        print()
        # ./mach.py --mode parameterization_find_args --path data/EEM/500/  --optimization_method local_minimization --data_dir asdfasdf -f
    """
    return objective_value


@jit(nopython=True, cache=True)
def lhsclassic(n, samples):
    #random.seed(0)
    cut = linspace(0, 1, samples + 1)
    u = random.rand(samples, n)
    for j in range(n):
        u[random.permutation(arange(samples)), j] = u[:, j] * cut[1] + cut[:samples]
    return u


class Parameterization:
    def __init__(self, sdf, ref_charges, method, optimization_method, minimization_method, num_of_samples, cpu, parameters, new_parameters, charges, data_dir, num_of_molecules, rewriting_with_force, subset_heuristic):
        start_time = date.now()
        control_existing_files(((sdf, True, "file"),
                                (ref_charges, True, "file"),
                                (parameters, True, "file"),
                                (data_dir, False, "directory")),
                               rewriting_with_force)
        mkdir(data_dir)
        method = getattr(import_module("modules.methods"), method)()
        method.load_parameters(parameters)
        set_of_molecules = SetOfMolecules(sdf, num_of_molecules)
        prepare_data_for_parameterization(set_of_molecules, ref_charges, method)

        print("Parameterizating of charges...")
        low_bond, high_bond = -0, 1
        bounds = [(low_bond, high_bond)] * len(method.parameters_values)
        if optimization_method == "local_minimization":
            _, final_parameters, course = local_minimization(method.parameters_values, minimization_method, method, set_of_molecules, bounds=bounds)
        elif optimization_method == "guided_minimization":
            samples = lhsclassic(len(method.parameters_values), samples=num_of_samples) * abs(high_bond - low_bond) + low_bond
            partial_f = partial(calculate_charges_and_statistical_data, method=method, set_of_molecules=set_of_molecules if not subset_heuristic else SubsetOfMolecules(set_of_molecules, method))
            with Pool(cpu) as pool:
                candidates_rmsd = list(pool.imap(partial_f, samples, chunksize=100))
            main_candidates = samples[list(map(candidates_rmsd.index, heapq.nsmallest(3 if cpu < 3 else cpu, candidates_rmsd)))]
            partial_f = partial(local_minimization, minimization_method=minimization_method, method=method, set_of_molecules=set_of_molecules, bounds=bounds)
            with Pool(cpu) as pool:
                parameters = [result for result in pool.map(partial_f, [parameters for parameters in main_candidates])]
            parameters.sort(key=itemgetter(0))
            course = [parameterization_data[2] for parameterization_data in parameters]
            final_parameters = parameters[0][1]
        method.new_parameters(final_parameters)
        method.calculate(set_of_molecules)
        print(colored("\033[Kok\n", "green"))

        atomic_types_charges, chg_molecules = prepare_data_for_comparison(method, set_of_molecules)
        comparison = Comparison(set_of_molecules, (method.results, atomic_types_charges, chg_molecules, charges), data_dir, rewriting_with_force, parameterization=method, course=[optimization_method, course])
        copyfile(sdf, path.join(data_dir, path.basename(sdf)))
        copyfile(ref_charges, path.join(data_dir, path.basename(ref_charges)))
        write_charges_to_file(path.join(data_dir, charges), method.results, set_of_molecules)
        summary_lines, parameters_lines = write_parameters_to_file(path.join(data_dir, new_parameters), method, set_of_molecules.file, optimization_method, minimization_method, start_time, num_of_samples if optimization_method == "guided_minimization" else None, cpu)
        comparison.write_html_parameterization(path.join(data_dir, "{}_{}.html".format(path.basename(sdf)[:-4], method)), path.basename(sdf), charges, path.basename(ref_charges), summary_lines, parameters_lines, new_parameters)


"""

num_of_samples = 100
data = ["{}_{}".format(str(method), len(method.parameters_values))]
for _ in range(5):
    print(num_of_samples)
    aaa = 0
    repete = 3
    for x in range(repete):
        samples = lhsclassic(len(method.parameters_values), samples=num_of_samples)
        manager = Manager()
        gm_course = manager.list()
        partial_f = partial(calculate_charges_and_statistical_data, method=method, set_of_molecules=set_of_molecules if not subset_heuristic else SubsetOfMolecules(set_of_molecules, method), course=gm_course)
        with Pool(cpu) as pool:
            candidates_rmsd = list(pool.imap(partial_f, samples, chunksize=100))
        mmm = min(candidates_rmsd)
        print(mmm)
        aaa += mmm
    data.append((aaa/repete, num_of_samples))
    num_of_samples *= 4
    print()
print()
from pprint import pprint ; pprint(data)
from sys import exit ; exit()

"""