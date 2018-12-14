from os.path import basename
from os import system
from glob import glob
from termcolor import colored
import git

def parameterization_meta(path, num_of_molecules, optimization_method, minimization_method, num_of_samples, cpu, RAM, walltime, subset_heuristic):  # only for my usage
    print("Control path...")
    sdf_files = glob("{}*.sdf".format(path))
    par_files = glob("{}*.par".format(path))
    chg_files = glob("{}*.chg".format(path))
    if len(par_files) != 1 or len(sdf_files) != 1 or len(chg_files) != 2:
        exit(colored("There is not 1 parameters file, 1 sdf file and 2 charges files in {}!".format(path), "red"))
    print(colored("ok\n", "green"))
    sdf_file = sdf_files[0]
    par_file = par_files[0]
    new_par_file = basename(par_file)
    method = par_file.split("_")[-2]
    ref_charges = "{}.chg".format(sdf_file[:-4])
    charges = "{}_{}.chg".format(basename(sdf_file)[:-4], method)
    command = "./mach.py --mode parameterization --method {} --optimization_method {} --minimization_method {} --parameters {} --sdf {} --ref_charges {} " \
              " --new_parameters {} --data_dir results_data --charges {} --cpu {} --git_hash {} " \
        .format(method, optimization_method, minimization_method, basename(par_file), basename(sdf_file),
                basename(ref_charges), new_par_file, charges, cpu, git.Repo(search_parent_directories=True).head.object.hexsha)
    command += " --num_of_molecules {}".format(num_of_molecules) if num_of_molecules else ""
    command += " --subset_heuristic " if subset_heuristic else ""
    command += " --num_of_samples {}".format(num_of_samples) if num_of_samples else ""
    system("./modules/parameterization_meta.sh {} {} {} '{} > /dev/null' {} {} {}".format(par_file, sdf_file, ref_charges, command, cpu,
                                                            RAM, walltime))


