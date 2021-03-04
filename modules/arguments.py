import argparse
import argcomplete
from termcolor import colored


def load_arguments():
    print("\nParsing arguments...")
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--mode",
                        help="Choose the MACH mode.",
                        choices=("calculation", "parameterization",
                                 "set_of_molecules_info", "comparison",
                                 "parameterization_meta"))
    parser.add_argument("--ats_types_pattern",
                        help="Use for comparison and set_of_molecules only. "
                             "Define used atomic classifier.",
                        choices=("plain", "hbo", "ba", "ba2"),
                        default="hbo")
    parser.add_argument("--emp_chgs_file",
                        help="File to store calculated charges or file with charges for comparison.")
    parser.add_argument("--chg_method",
                        help="Empirical method for calculation or parameterization partial atomic charges.",
                        choices=("EEM", "EQEq", "SQE", "SQEq0", "SQEqp", "SQEqpc", "QEq_Louwen_Vogt",
                                 "QEq_Nisimoto_Mataga", "QEq_Nisimoto_Mataga_Wiess", "QEq_Ohno",
                                 "QEq_Ohno_Klopman", "QEq_Dasgupta_Huzinaga"))
    parser.add_argument("--maxiter",
                        help="Use for local minimization only. "
                             "Define maximum number of iterations.",
                        default=10000000,
                        type=int)
    parser.add_argument("--num_of_candidates",
                        help="Use for optGM optimization method only. "
                             "Define number of used candidates.",
                        default=3,
                        type=int)
    parser.add_argument("--num_of_samples",
                        help="Use for optGM optimization method only. "
                             "Define number of used initial samples.",
                        default=10000,
                        type=int)
    parser.add_argument("--min_subset",
                        help="Use for optGM optimization method only. "
                             "Minimal subset of molecules that contains n atoms of each atom type "
                             "is used for parameterization.",
                        default=1,
                        type=int)
    parser.add_argument("--optimization_method",
                        help="Optimization method for parameterization.",
                        choices=("local_minimization", "optGM", "GDMIN"),
                        default="optGM")
    parser.add_argument("--subset",
                        help="Use for optGM optimization method only. "
                             "Subset of molecules that contains n atoms of each atom type "
                             "is used for parameterization.",
                        default=10,
                        type=int)
    parser.add_argument("--params_file",
                        help="File with parameters.")
    parser.add_argument("--percent_par",
                        help="Set how many percent of set of molecules should be used for parameterization.",
                        default=80,
                        type=int)
    parser.add_argument("--RAM",
                        help="Use for parameterization_meta mode only. "
                             "Define maximum RAM usage for META job in GB.",
                        default=10,
                        type=int)
    parser.add_argument("--random_seed",
                        help="Use for parameterization mode only. "
                             "Set initial random state to guarantee reproduction of charges. "
                             "Set 0 for fully random parameterization.",
                        default=1,
                        type=int)
    parser.add_argument("--ref_chgs_file",
                        help="File with reference charges.")
    parser.add_argument("--sdf_file",
                        help="Sdf file with molecules.")
    parser.add_argument("--walltime",
                        help="Use for parameterization_meta mode only. "
                             "Define maximum time for META job in hours.",
                        default=100,
                        type=int)

    argcomplete.autocomplete(parser)
    args = parser.parse_args()

    if args.mode == "set_of_molecules_info":
        if args.sdf_file is None:
            parser.error("For set_of_molecules_info mode choose --sdf_file.")

    elif args.mode == "calculation":
        if any(arg is None for arg in [args.sdf_file,
                                       args.chg_method,
                                       args.params_file]):
            parser.error("For calculation mode choose --sdf_file, "
                         "--chgs_file and --params_file.")

    elif args.mode == "comparison":
        if any(arg is None for arg in [args.sdf_file,
                                       args.ref_chgs_file,
                                       args.emp_chgs_file]):
            parser.error("For comparison mode choose --sdf_file, "
                         "--emp_chgs_file and --ref_chgs_file.")

    elif args.mode in ["parameterization", "parameterization_meta"]:
        if any(arg is None for arg in [args.sdf_file,
                                       args.ref_chgs_file,
                                       args.chg_method]):
            parser.error("For parameterization mode choose --sdf_file, "
                         "--ref_chgs_file and --chg_method.")

    print(colored("ok\n", "green"))
    return args
