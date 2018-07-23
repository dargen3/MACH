import argparse
import argcomplete


def load_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", help="Choice what MACH should do.",
                        required=True,
                        choices=("set_of_molecules_info", "calculation", "parameterization", "comparison"))
    parser.add_argument("--sdf", help="Sdf file with molecules data.")
    parser.add_argument("--charges", help="File to store calculated charges of file with charges for comparison.")
    parser.add_argument("--ref_charges", help="File with reference charges for comparison.")
    parser.add_argument("--parameters", help="File with parameters.")
    # metody?? jake
    parser.add_argument("--method", help="Empirical method for calculation partial atomic charges.",
                        choices=("EEM", "SFKEEM", "QEq"))
    parser.add_argument("--atomic_types_pattern",
                        help="For mode set_of_molecules_info only. Define atomic types for statistics",
                        choices=("atom", "atom_high_bond"))
    parser.add_argument("-f", "--rewriting_with_force", action="store_true",
                        help="All existed files with the same names like your outputs will be replaced.")
    argcomplete.autocomplete(parser)
    args = parser.parse_args()

    if args.mode == "set_of_molecules_info":
        if args.sdf is None or args.atomic_types_pattern is None:
            parser.error("For set_of_molecules_info must choice --sdf and --atomic_types_pattern.")

    if args.mode == "calculation":
        if args.method is None or args.sdf is None or args.parameters is None or args.charges is None:
            parser.error("For calculation must be choice --method, --sdf, --parameters and --charges.")

    elif args.mode == "parameterization":
        if args.charges_for_parameterization is None or args.method is None \
                or args.sdf is None or args.parameters is None \
                or args.new_parameters is None or args.charges is None:
            parser.error("For parameterization must be choice --right_charges, --method, --sdf_input,",
                         "--parameters, --chg_output and --new_parameters.")

    if args.mode == "comparison":
        if args.charges is None or args.ref_charges is None:
            parser.error("For comparison must choice --charges and --ref_charges.")

    return args
