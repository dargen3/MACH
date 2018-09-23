import argparse
import argcomplete


def load_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", help="Choice what MACH should do.",
                        required=True,
                        choices=("set_of_molecules_info", "calculation", "parameterization", "comparison", "parameterization_find_args",
                                 "calculation_meta", "parameterization_meta", "make_complete_html"))  # only for my usage
    parser.add_argument("--sdf", help="Sdf file with molecules data.")
    parser.add_argument("--charges", help="File to store calculated charges of file with charges for comparison.")
    parser.add_argument("--ref_charges", help="File with reference charges for comparison.")
    parser.add_argument("--parameters", help="File with parameters.")
    parser.add_argument("--new_parameters", help="File to store new parametes.")
    # metody?? jake
    parser.add_argument("--method", help="Empirical method for calculation partial atomic charges.",
                        choices=("EEM", "SFKEEM", "QEq", "GM"))
    parser.add_argument("--optimization_method", help="Optimization method for parameterization.", choices=("minimization", "guided_minimization", "differential_evolution", "CRS"))
    parser.add_argument("--cpu", help="Only for optimization method guided minimization. Define number of used cpu for parameterization.", default=1, type=int)
    parser.add_argument("--path", help="Only for parameterization_find_args. Define path to files.")
    parser.add_argument("--data_dir", help="For parameterization and comparison only. Defined directory is created to store correlation graphs and html file.")
    parser.add_argument("--RAM", help="For parameterization_meta and calculation_meta only. Define max. RAM usage for META job in GB.", default=2, type=int)  # only for my usage
    parser.add_argument("--walltime", help="For parameterization_meta and calculation_meta only. Define max time for META job in hours.", default=2, type=int)  # only for my usage
    parser.add_argument("--atomic_types_pattern",
                        help="For mode set_of_molecules_info only. Define atomic types for statistics",
                        choices=("atom", "atom_high_bond"), default="atom_high_bond")
    parser.add_argument("--num_of_molecules", help="Only these number of molecules will be loaded.", type=int)
    parser.add_argument("-f", "--rewriting_with_force", action="store_true",
                        help="All existed files with the same names like your outputs will be replaced.")
    argcomplete.autocomplete(parser)
    args = parser.parse_args()

    if args.mode == "set_of_molecules_info":
        if args.sdf is None:
            parser.error("For set_of_molecules_info must choice --sdf.")

    if args.mode == "calculation":
        if args.method is None or args.sdf is None or args.parameters is None or args.charges is None:
            parser.error("For calculation must be choice --method, --sdf, --parameters and --charges.")

    elif args.mode == "parameterization":
        if args.ref_charges is None or args.method is None \
                or args.sdf is None or args.parameters is None \
                or args.new_parameters is None or args.charges is None \
                or args.optimization_method is None or args.data_dir is None:
            parser.error("For parameterization must be choice --ref_charges, --method, --sdf, "
                         "--parameters, --new_parameters, --optimization_method, --data_dir and --charges.")

    if args.mode == "parameterization_find_args":
        if args.path is None or args.optimization_method is None or args.data_dir is None:
            parser.error("For parameterization_find_args must choice --path, --data_dir and --optimization_method.")

    if args.mode == "comparison":
        if args.charges is None or args.ref_charges is None or args.data_dir is None:
            parser.error("For comparison must choice --charges, --ref_charges and --data_dir.")

    if args.mode == "calculation_meta":  # only for my usage
        if args.method is None or args.sdf is None or args.parameters is None or args.charges is None:
            parser.error("For calculation_meta must be choice --method, --sdf, --parameters and --charges.")

    if args.mode == "parameterization_meta":
        if args.path is None or args.optimization_method is None:
            parser.error("For parameterization_meta must choice --path and --optimization_method.")


    return args
