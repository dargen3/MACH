from os.path import basename
from os import system
from glob import glob
from termcolor import colored
import git


def parameterization_meta(sdf_file, ref_charges, parameters, method, optimization_method, minimization_method, atomic_types_pattern, num_of_samples, num_of_candidates, parameterization_subset, cpu, RAM, walltime, random_seed, convert_parameters):
    if not parameters:
        parameters = "modules/parameters/{}.json".format(method)
    command = "/storage/praha1/home/dargen3/miniconda3/bin/python3.7 mach.py --mode parameterization --method {} --optimization_method {} --minimization_method {} --parameters {} --sdf_file {} --ref_chg_file {} " \
              " --data_dir results_data --cpu {} --git_hash {} --atomic_types_pattern {} --num_of_samples {} --parameterization_subset {} --num_of_candidates {} --random_seed {}" \
              .format(method, optimization_method, minimization_method, basename(parameters), basename(sdf_file),
                      basename(ref_charges), cpu, git.Repo(search_parent_directories=True).head.object.hexsha,
                      atomic_types_pattern, num_of_samples, parameterization_subset, num_of_candidates, random_seed)
    command += " --convert_parameters " if convert_parameters else ""
    system("./modules/parameterization_meta.sh {} {} {} '{}' {} {} {}".format(parameters, sdf_file, ref_charges, command, cpu,
                                                                              RAM, walltime))
