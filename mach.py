#!/usr/bin/env python3
from termcolor import colored
import warnings
warnings.filterwarnings("ignore")
from modules.arguments import load_arguments
from modules.set_of_molecules import SetOfMolecules
from modules.calculation import Calculation
from modules.comparison import Comparison
from modules.parameterization import Parameterization
from modules.parameterization_find_args import parameterization_find_args  # only for my usage
from modules.calculation_meta import calculation_meta  # only for my usage
from modules.parameterization_meta import parameterization_meta
from modules.make_complete_html import make_complete_html

if __name__ == '__main__':
    args = load_arguments()
    print(colored("\nMACH is running with mode: {}\n".format(args.mode), "blue"))
    if args.mode == "set_of_molecules_info":
        set_of_molecules = SetOfMolecules(args.sdf, args.num_of_molecules)
        set_of_molecules.info(args.atomic_types_pattern)

    if args.mode == "calculation":
        Calculation(args.sdf,
                    args.method,
                    args.parameters,
                    args.charges,
                    args.rewriting_with_force)

    if args.mode == "parameterization":
        Parameterization(args.sdf,
                         args.ref_charges,
                         args.method,
                         args.optimization_method,
                         args.minimization_method,
                         args.GM_level,
                         args.cpu,
                         args.parameters,
                         args.new_parameters,
                         args.charges,
                         args.data_dir,
                         args.num_of_molecules,
                         args.rewriting_with_force)

    if args.mode == "comparison":
        Comparison(args.ref_charges,
                   args.charges,
                   args.data_dir,
                   args.rewriting_with_force)

    if args.mode == "parameterization_find_args": # only for my usage
        parameterization_find_args(args.path,
                                   args.optimization_method,
                                   args.minimization_method,
                                   args.GM_level,
                                   args.cpu,
                                   args.data_dir,
                                   args.num_of_molecules,
                                   args.rewriting_with_force)


    if args.mode == "calculation_meta":  # only for my usage
        calculation_meta(args.sdf,
                         args.method,
                         args.parameters,
                         args.charges,
                         args.RAM,
                         args.walltime)

    if args.mode == "parameterization_meta":  # only for my usage
        parameterization_meta(args.path,
                              args.optimization_method,
                              args.minimization_method,
                              args.GM_level,
                              args.cpu,
                              args.RAM,
                              args.walltime)

    if args.mode == "make_complete_html":
        make_complete_html()
