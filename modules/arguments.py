import argparse

import argcomplete
from termcolor import colored


def load_arguments():
    print("Parsing arguments...")
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser._action_groups.pop()
    required = parser.add_argument_group('Required arguments')
    optional = parser.add_argument_group('Optional arguments')
    required.add_argument("--mode",
                          help="Choose the MACH mode.",
                          required=True,
                          choices=("calculation", "parameterization",
                                   "comparison", "parameterization_meta"))
    optional.add_argument("--atomic_types_pattern",
                          help="Use for parameterization only. Argument defines used atomic classifier.",
                          choices=("plain", "hbo", "plain-ba", "plain-ba-sb"))
    optional.add_argument("--cpu",
                          help="Use for \"guided minimization\" optimization method only. Define number of used CPU for parameterization.",
                          default=1,
                          type=int)
    optional.add_argument("--data_dir",
                          help="Use for parameterization and comparison modes only. Defined directory stores all computed data.")
    optional.add_argument("--emp_chg_file",
                          help="File to store calculated charges or file with charges for comparison.")
    optional.add_argument("-f", "--rewriting_with_force",
                          help="All MACH output files and directories will be rewritten.",
                          action="store_true")
    optional.add_argument("--git_hash",
                          help="For internal usage only.")
    optional.add_argument("--chg_method",
                          help="Empirical method for calculation or parameterization partial atomic charges.",
                          choices=("EEM", "QEq", "SQE", "EQEq", "EQEqc"))
    optional.add_argument("--minimization_method",
                          help="Minimization method for parameterization.",
                          default="SLSQP")
    optional.add_argument("--num_of_candidates",
                          help="Use for \"guided minimization\" optimization method only. Define number of used candidates.",
                          default=30,
                          type=int)
    optional.add_argument("--num_of_samples",
                          help="Use for \"guided minimization\" optimization method only. Define number of used initial samples.",
                          default=500000,
                          type=int)
    optional.add_argument("--optimization_method",
                          help="Optimization method for parameterization.",
                          choices=("local_minimization", "guided_minimization"),
                          default="guided_minimization")
    optional.add_argument("--parameterization_subset",
                          help="Use for parameterization mode only. Minimal subset of molecules that contains n atoms of each atom type is used for parameterization. Other molecules are used for validation.",
                          default=100,
                          type=int)
    optional.add_argument("--par_file",
                          help="File with parameters.")
    optional.add_argument("--RAM",
                          help="Use for parameterization_meta and calculation_meta modes only. Define maximum RAM usage for META job in GB.",
                          default=10,
                          type=int)
    optional.add_argument("--random_seed",
                          help="Use for parameterization mode only. Set initial random state to guarantee reproduction of results. Set 0 for full random parameterization.",
                          default=1,
                          type=int)
    optional.add_argument("--ref_chg_file",
                          help="File with reference charges.")
    optional.add_argument("--sdf_file",
                          help="Sdf file with molecules.")
    optional.add_argument("--walltime",
                          help="Use for parameterization_meta and calculation_meta modes only. Define maximum time for META job in hours.",
                          default=600,
                          type=int)

    argcomplete.autocomplete(parser)
    args = parser.parse_args()

    if args.mode == "calculation":
        if any(arg is None for arg in [args.chg_method,
                                       args.sdf_file,
                                       args.par_file,
                                       args.emp_chg_file]):
            parser.error("For calculation mode choose --chg_method, --sdf_file, --par_file and --emp_chg_file.")
        elif args.atomic_types_pattern is not None:
            parser.error("Combination of --mode calculation and argument --atomic_types_pattern is not allowed! Atomic types pattern is set by parameters file.")


    # elif args.mode == "parameterization":
    #     if args.ref_chg_file is None \
    #             or args.method is None \
    #             or args.sdf_file is None \
    #             or args.par_file in None \
    #             or args.data_dir is None:
    #         parser.error("For parameterization mode choose --ref_chg_file, --chg_method, --sdf_file, --par_file and --data_dir")

    # elif args.mode == "comparison":
    #     if args.emp_chg_file is None or args.ref_chg_file is None or args.data_dir is None:
    #         parser.error("For comparison mode choose --emp_chg_file, --ref_chg_file and --data_dir.")
    #
    # elif args.mode == "calculation_meta":
    #     if args.method is None or args.sdf_file is None or args.emp_chg_file is None:
    #         parser.error("For calculation_meta mode choose --chg_method, --sdf_file and --emp_chg_file.")
    #
    # elif args.mode == "parameterization_meta":
    #     if args.ref_chg_file is None or args.method is None or args.sdf_file is None:
    #         parser.error("For parameterization_meta mode choose --ref_chg_file, --chg_method and --sdf_file. ")
    #
    # if args.parameterization_subset < 1:
    #     parser.error("Error! parameterization_subset value must be higher then 0!")

    print(colored("ok\n", "green"))
    return args
