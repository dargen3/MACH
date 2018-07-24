#!/usr/bin/env python3
from modules.arguments import load_arguments
from modules.set_of_molecules import SetOfMolecules
from modules.calculation import calculate_charges
from modules.comparison import Compare

from termcolor import colored

if __name__ == '__main__':
    args = load_arguments()
    print(colored("\nMACH is running with mode: {}\n".format(args.mode), "blue"))

    if args.mode == "set_of_molecules_info":
        set_of_molecules = SetOfMolecules(args.sdf)
        set_of_molecules.info(args.atomic_types_pattern)

    if args.mode == "calculation":
        calculate_charges(args.sdf,
                          args.method,
                          args.parameters,
                          args.charges,
                          args.rewriting_with_force)

    if args.mode == "comparison":
        Compare(args.ref_charges,
                args.charges,
                args.save_fig,
                from_file=True)

