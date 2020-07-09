import argparse
from datetime import datetime

import argcomplete
import git
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
                        choices=("plain", "hbo", "plain-ba", "plain-ba-sb"),
                        default="plain")
    parser.add_argument("--cpu",
                        help="Use for \"guided minimization\" optimization method only. "
                             "Define number of used CPU for parameterization.",
                        default=1,
                        type=int)
    parser.add_argument("--data_dir",
                        help="Defined directory stores all computed data.",
                        default=None)
    parser.add_argument("--emp_chgs_file",
                        help="File to store calculated charges or file with charges for comparison.")
    parser.add_argument("-f", "--rewriting_with_force",
                        help="All MACH output files and directories will be rewritten.",
                        action="store_true")
    parser.add_argument("--git_hash",
                        help="For internal usage only.")
    parser.add_argument("--chg_method",
                        help="Empirical method for calculation or parameterization partial atomic charges.",
                        choices=("EEM", "QEq", "SQE", "EQEq", "EQEqc"))
    parser.add_argument("--num_of_candidates",
                        help="Use for \"guided minimization\" optimization method only. "
                             "Define number of used candidates.",
                        default=30,
                        type=int)
    parser.add_argument("--num_of_samples",
                        help="Use for \"guided minimization\" optimization method only. "
                             "Define number of used initial samples.",
                        default=500000,
                        type=int)
    parser.add_argument("--optimization_method",
                        help="Optimization method for parameterization.",
                        choices=("local_minimization", "guided_minimization", "opt_guided_minimization", "differential_evolution"),
                        default="opt_guided_minimization")
    parser.add_argument("--subset",
                        help="Use for parameterization mode only. "
                             "Minimal subset of molecules that contains n atoms of each atom type "
                             "is used for parameterization. Other molecules are used for validation.",
                        default=100,
                        type=int)
    parser.add_argument("--params_file",
                        help="File with parameters.")
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
                        default=24,
                        type=int)

    argcomplete.autocomplete(parser)
    args = parser.parse_args()

    if not args.data_dir:
        args.data_dir = f"{datetime.now().strftime('%Y-%m-%d-%H-%M-%S')}_{args.mode}_{args.chg_method}_{args.sdf_file.split('/')[-1][:-4]}"

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
            parser.error("For comparison mode choose --sdf_file, --emp_chgs_file and --ref_chgs_file.")

    elif args.mode in ["parameterization", "parameterization_meta"]:
        if any(arg is None for arg in [args.sdf_file,
                                       args.ref_chgs_file,
                                       args.chg_method,
                                       args.params_file]):
            parser.error("For parameterization mode choose --sdf_file, "
                         "--ref_chgs_file, --chg_method and --params_file.")

        if not args.git_hash:
            args.git_hash = git.Repo(search_parent_directories=True).head.object.hexsha

    if args.subset < 1:
        parser.error("Error! subset value must be higher then 0!")

    print(colored("ok\n", "green"))
    return args
