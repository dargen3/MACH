#!/usr/bin/env python3
from modules.arguments import load_arguments
from modules.calculation import Calculation
from modules.calculation_cutoff import CalculationCutoff
from modules.calculation_meta import calculation_meta
from modules.clusterization import clusterize
from modules.comparison import Comparison
from modules.parameterization import Parameterization
from modules.parameterization_meta import parameterization_meta
from modules.set_of_molecules import SetOfMolecules

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
    print(colored("\nMACH is running with mode: {}\n".format(args.mode), "blue"))
    if args.mode == "calculation":
        Calculation(args.sdf_file,
                    args.method,
                    args.parameters,
                    args.emp_chg_file,
                    args.atomic_types_pattern,
                    args.rewriting_with_force)

    elif args.mode == "calculation_cutoff":
        CalculationCutoff(args.sdf_file,
                    args.method,
                    args.parameters,
                    args.emp_chg_file,
                    args.atomic_types_pattern,
                    args.rewriting_with_force)

    elif args.mode == "parameterization":
        Parameterization(args.sdf_file,
                         args.ref_chg_file,
                         args.parameters,
                         args.method,
                         args.optimization_method,
                         args.minimization_method,
                         args.atomic_types_pattern,
                         args.num_of_molecules,
                         args.num_of_samples,
                         args.num_of_candidates,
                         args.subset_heuristic,
                         args.validation,
                         args.cpu,
                         args.data_dir,
                         args.rewriting_with_force,
                         args.random_seed,
                         args.git_hash)

    elif args.mode == "comparison":
        Comparison().comparison(args.ref_chg_file,
                                args.emp_chg_file,
                                args.data_dir,
                                args.rewriting_with_force)

    elif args.mode == "calculation_meta":
        calculation_meta(args.sdf_file,
                         args.method,
                         args.parameters,
                         args.emp_chg_file,
                         args.atomic_types_pattern,
                         args.RAM,
                         args.walltime)

    elif args.mode == "parameterization_meta":
        parameterization_meta(args.sdf_file,
                              args.ref_chg_file,
                              args.parameters,
                              args.method,
                              args.optimization_method,
                              args.minimization_method,
                              args.atomic_types_pattern,
                              args.num_of_molecules,
                              args.num_of_samples,
                              args.num_of_candidates,
                              args.subset_heuristic,
                              args.validation,
                              args.cpu,
                              args.RAM,
                              args.walltime,
                              args.random_seed)

    elif args.mode == "clusterization":
        clusterize(args.ref_chg_file, args.sdf_file, args.atomic_types_pattern)
