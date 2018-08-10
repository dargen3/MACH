#!/usr/bin/env python3
from termcolor import colored
import warnings
warnings.filterwarnings("ignore")
from modules.arguments import load_arguments
from modules.set_of_molecules import SetOfMolecules
from modules.calculation import Calculation
from modules.comparison import Comparison
from modules.parameterization import Parameterization


if __name__ == '__main__':
    args = load_arguments()
    print(colored("\nMACH is running with mode: {}\n".format(args.mode), "blue"))
    if args.mode == "set_of_molecules_info":
        set_of_molecules = SetOfMolecules(args.sdf)
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
                         args.cpu,
                         args.parameters,
                         args.new_parameters,
                         args.charges,
                         args.rewriting_with_force,
                         args.save_fig)

    if args.mode == "comparison":
        Comparison(args.ref_charges,
                   args.charges,
                   args.save_fig,
                   from_file=True)


