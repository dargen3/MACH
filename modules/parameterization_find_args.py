from .parameterization import Parameterization
from glob import glob
from sys import exit
from os.path import basename
from termcolor import colored
from json import load

def parameterization_find_args(path, optimization_method, minimization_method, num_of_samples, cpu, data_dir, num_of_molecules, rewriting_with_force, subset_heuristic):  # only for my usage
    print("Control path...")
    sdf_files = glob("{}*.sdf".format(path))
    par_files = glob("{}parameters.json".format(path))
    chg_files = glob("{}*.chg".format(path))
    if len(par_files) != 1 or len(sdf_files) != 1 or len(chg_files) != 2:
        exit(colored("There is not 1 parameters.json file, 1 sdf file and 2 charges files in {}!".format(path), "red"))
    print(colored("ok\n", "green"))
    sdf_file = sdf_files[0]
    par_file = par_files[0]
    method = load(open(par_files[0]))["metadata"]["method"]
    ref_charges = "{}.chg".format(sdf_file[:-4])
    Parameterization(sdf_file,
                     ref_charges,
                     method,
                     optimization_method,
                     minimization_method,
                     num_of_samples,
                     cpu,
                     par_file,
                     data_dir,
                     num_of_molecules,
                     rewriting_with_force,
                     subset_heuristic)