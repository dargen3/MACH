#!/usr/bin/env python3
from modules.arguments import load_arguments
from modules.calculation import calculate_chgs
from modules.comparison import comparison
from modules.parameterization import parameterize
from modules.parameterization_meta import parameterization_meta
from modules.set_of_molecules import set_of_mols_info

from numba import jit
from numpy import random
from termcolor import colored
from warnings import filterwarnings

filterwarnings("ignore")

@jit(nopython=True, cache=True)
def numba_seed(seed):
    random.seed(seed)


if __name__ == '__main__':
    args = load_arguments()
    if args.random_seed != 0:
        random.seed(args.random_seed)
        numba_seed(args.random_seed)
    print(colored(f"MACH is running with mode: {args.mode}\n", "blue"))

    if args.mode == "set_of_molecules_info":
        set_of_mols_info(args.sdf_file,
                         args.ats_types_pattern)

    elif args.mode == "calculation":
        calculate_chgs(args.sdf_file,
                       args.chg_method,
                       args.params_file,
                       args.data_dir,
                       args.rewriting_with_force)


    elif args.mode == "parameterization":
        parameterize(args.sdf_file,
                     args.ref_chgs_file,
                     args.chg_method,
                     args.params_file,
                     args.optimization_method,
                     args.minimization_method,
                     args.num_of_samples,
                     args.num_of_candidates,
                     args.par_subset,
                     args.data_dir,
                     args.rewriting_with_force,
                     args.git_hash)

    elif args.mode == "comparison":
        comparison(args.sdf_file,
                   args.ref_chgs_file,
                   args.emp_chgs_file,
                   args.ats_types_pattern,
                   args.data_dir,
                   args.rewriting_with_force)



    elif args.mode == "parameterization_meta":
        parameterization_meta(args.sdf_file,
                              args.ref_chgs_file,
                              args.chg_method,
                              args.params_file,
                              args.optimization_method,
                              args.minimization_method,
                              args.num_of_samples,
                              args.num_of_candidates,
                              args.par_subset,
                              args.cpu,
                              args.RAM,
                              args.walltime,
                              args.random_seed,
                              args.data_dir)

