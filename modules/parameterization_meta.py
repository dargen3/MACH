from os.path import basename
from os import system
from glob import glob
from termcolor import colored
import git


def parameterization_meta(sdf_file, ref_charges, parameters, method, optimization_method, minimization_method, atomic_types_pattern, num_of_molecules, num_of_samples, num_of_candidates, subset_heuristic, parameterization_subset, cpu, RAM, walltime, random_seed):
    if not parameters:
        parameters = "modules/parameters/{}.json".format(method)
    command = "./mach.py --mode parameterization --method {} --optimization_method {} --minimization_method {} --parameters {} --sdf_file {} --ref_chg_file {} " \
              " --data_dir results_data --cpu {} --git_hash {} --atomic_types_pattern {} --subset_heuristic {} --num_of_samples {} --parameterization_subset {} --num_of_candidates {} --random_seed {}" \
              .format(method, optimization_method, minimization_method, basename(parameters), basename(sdf_file),
                      basename(ref_charges), cpu, git.Repo(search_parent_directories=True).head.object.hexsha,
                      atomic_types_pattern, subset_heuristic, num_of_samples, parameterization_subset, num_of_candidates, random_seed)
    command += " --num_of_molecules {} ".format(num_of_molecules) if num_of_molecules else ""
    system("./modules/parameterization_meta.sh {} {} {} '{}' {} {} {}".format(parameters, sdf_file, ref_charges, command, cpu,
                                                                              RAM, walltime))
