from .parameterization import Parameterization
from glob import glob
from sys import exit
from os.path import basename
from termcolor import colored
from json import load

def parameterization_find_args(path, optimization_method, minimization_method, num_of_samples, cpu, data_dir, num_of_molecules, rewriting_with_force, subset_heuristic, atomic_types_pattern):  # only for my usage
    print("Control path...")
    sdf_files = glob("{}*.sdf".format(path))
    chg_files = glob("{}*.chg".format(path))
    if len(sdf_files) != 1 or len(chg_files) != 2:
        exit(colored("There is not 1 parameters.json file, 1 sdf file and 2 charges files in {}!".format(path), "red"))
    print(colored("ok\n", "green"))
    sdf_file = sdf_files[0]
    for charges_file in chg_files:
        if len(basename(charges_file).split("_")) == 2:
            method = basename(charges_file).split("_")[1].split(".")[0]
            break
    ref_charges = "{}.chg".format(sdf_file[:-4])
    Parameterization(sdf_file,
                     ref_charges,
                     method,
                     optimization_method,
                     minimization_method,
                     num_of_samples,
                     cpu,
                     None,
                     data_dir,
                     num_of_molecules,
                     rewriting_with_force,
                     subset_heuristic,
                     atomic_types_pattern)