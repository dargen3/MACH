import argparse
import argcomplete


def load_arguments():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--mode", help="Choose the MACH mode.",
                        required=True,
                        choices=("set_of_molecules_info", "calculation", "calculation_cutoff", "parameterization", "comparison",
                                 "calculation_meta", "parameterization_meta", "clusterization"))
    parser.add_argument("--sdf_file", help="Sdf file with molecules.")
    parser.add_argument("--emp_chg_file", help="File to store calculated charges or file with charges for comparison.")
    parser.add_argument("--ref_chg_file", help="File with reference charges.")
    parser.add_argument("--parameters", help="File with parameters.")
    parser.add_argument("--method", help="Empirical method for calculation or parameterization partial atomic charges.",
                        choices=("EEM", "SFKEEM", "QEq", "PEOE", "MGC", "ACKS2", "COMBA", "DENR"))
    parser.add_argument("--optimization_method", help="Optimization method for parameterization.", choices=("local_minimization", "guided_minimization"), default="guided_minimization")
    parser.add_argument("--minimization_method", help="Minimization method for parameterization.", default="SLSQP")
    parser.add_argument("--cpu", help="Use for \"guided minimization\" optimization method only. Define number of used CPU for parameterization.", default=1, type=int)
    parser.add_argument("--num_of_samples", help="Use for \"guided minimization\" optimization method only. Define number of used initial samples.", default=500000, type=int)
    parser.add_argument("--num_of_candidates", help="Use for \"guided minimization\" optimization method only. Define number of used candidates.", default=30, type=int)
    parser.add_argument("--subset_heuristic", help="Use for \"guided minimization\" optimization method only. Minimal subset of molecules that contains n atoms of each atom type is used for first step of \"guided minimization\". If 0 is set, full set of molecules is used. Less value than 5 is not recommended.", default=5, type=int)
    parser.add_argument("--random_seed", help="Use for parameterization mode only. Set initial random state to guarantee reproduction of results. Set 0 for full random parameterization.", type=int, default=1)
    parser.add_argument("--data_dir", help="Use for parameterization and comparison modes only. Defined directory stores all computed data.")
    parser.add_argument("--RAM", help="Use for parameterization_meta and calculation_meta modes only. Define maximum RAM usage for META job in GB.", default=10, type=int)
    parser.add_argument("--walltime", help="Use for parameterization_meta and calculation_meta modes only. Define maximum time for META job in hours.", default=10, type=int)
    parser.add_argument("--git_hash", help="For internal usage only.")
    parser.add_argument("--validation", help="Define how many percent of set of molecules will be used for validation. Use 100 for maximum validation set.", default=10, type=int)
    parser.add_argument("--atomic_types_pattern",
                        help="Use for set_of_molecules_info and parameterization only. Argument defines used atomic classifier.",
                        choices=("plain", "hbo", "hbob"), default="hbo")  # hbob
    parser.add_argument("--num_of_molecules", help="Number of molecules loaded.", type=int)
    parser.add_argument("-f", "--rewriting_with_force", action="store_true",
                        help="All MACH output files and directories will be rewritten.")
    argcomplete.autocomplete(parser)
    args = parser.parse_args()

    if args.mode == "set_of_molecules_info":
        if args.sdf_file is None:
            parser.error("For set_of_molecules_info mode choose --sdf_file.")

    elif args.mode == "calculation":
        if args.method is None or args.sdf_file is None or args.emp_chg_file is None:
            parser.error("For calculation mode choose --method, --sdf_file and --emp_chg_file.")

    elif args.mode == "calculation":
        if args.method is None or args.sdf_file is None or args.emp_chg_file is None:
            parser.error("For calculation_cutoff mode choose --method, --sdf_file and --emp_chg_file.")

    elif args.mode == "parameterization":
        if args.ref_chg_file is None or args.method is None \
                or args.sdf_file is None or args.data_dir is None:
            parser.error("For parameterization mode choose --ref_chg_file, --method, --sdf_file and --data_dir")

    elif args.mode == "comparison":
        if args.emp_chg_file is None or args.ref_chg_file is None or args.data_dir is None:
            parser.error("For comparison mode choose --emp_chg_file, --ref_chg_file and --data_dir.")

    elif args.mode == "calculation_meta":
        if args.method is None or args.sdf_file is None or args.emp_chg_file is None:
            parser.error("For calculation_meta mode choose --method, --sdf_file and --emp_chg_file.")

    elif args.mode == "parameterization_meta":
        if args.ref_chg_file is None or args.method is None or args.sdf_file is None:
            parser.error("For parameterization_meta mode choose --ref_chg_file, --method and --sdf_file. ")

    elif args.mode == "clusterization":
        if args.ref_chg_file is None or args.sdf_file is None:
            parser.error("For clusterization mode choose --ref_chg_file and --sdf_file.")

    if args.mode in ["parameterization", "parameterization_meta"] and type(args.num_of_molecules) == int and args.num_of_molecules < 2:
        parser.error("Error! There must be more then 1 molecule for parameterization!")

    if not 101 > args.validation > 0:
        parser.error("Error! 1%-99% of set of molecules have to be used to validation! 100 is used for maximum validation set.")

    return args
