from .parameterization import Parameterization
from glob import glob
from sys import exit
from termcolor import colored

def parameterization_find_args(path, optimization_method, cpu, rewriting_with_force, save_fig):
    print("Control path...")
    sdf_files = glob("{}*.sdf".format(path))
    par_files = glob("{}*.par".format(path))
    chg_files = glob("{}*.chg".format(path))
    if len(par_files) != 1 or len(sdf_files) != 1 or len(chg_files) != 2:
        exit(colored("There is not 1 parameters file, 1 sdf file and 2 charges files in {}!".format(path), "red"))
    print(colored("ok\n", "green"))
    sdf_file = sdf_files[0]
    par_file = par_files[0]
    new_par_file = par_file.split("/")[-1]
    method = par_file.split("_")[-2]
    ref_charges = "{}.chg".format(sdf_file[:-4])
    charges = "{}_{}.chg".format(sdf_file.split("/")[-1][:-4], method)
    Parameterization(sdf_file,
                     ref_charges,
                     method,
                     optimization_method,
                     cpu,
                     par_file,
                     new_par_file,
                     charges,
                     rewriting_with_force,
                     save_fig)
