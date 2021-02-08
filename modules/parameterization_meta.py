from os import system
from os.path import basename

import git
from termcolor import colored


def parameterization_meta(sdf_file: str,
                          ref_chgs_file: str,
                          chg_method: str,
                          params_file: str,
                          ats_types_pattern: str,
                          percent_par: int,
                          optimization_method: str,
                          num_of_samples: int,
                          num_of_candidates: int,
                          subset: int,
                          cpu: int,
                          RAM: int,
                          walltime: int,
                          random_seed: int,
                          data_dir: str):

    print("Copying of data to MetaCentrum...")
    system(f"ssh dargen3@nympha.metacentrum.cz "
           f"'cd /storage/praha1/home/dargen3/mach; "
           f"mkdir {data_dir}; "
           f"mkdir {data_dir}/input_files; "
           f"mkdir {data_dir}/source_code; "
           f"mkdir {data_dir}/output_files; "
           f"cp structures/{basename(sdf_file)} {data_dir}/input_files ;"
           f"cp reference_charges/{basename(ref_chgs_file)} {data_dir}/input_files '")

    # předělat!!!!! 
    system(f"scp {sdf_file} {ref_chgs_file} {params_file} "
           f"dargen3@nympha.metacentrum.cz:/storage/praha1/home/dargen3/mach/{data_dir}/input_files")


    print(colored("ok\n", "green"))

    command = f" /storage/praha1/home/dargen3/miniconda3/bin/python3.8 mach.py " \
              f" --mode parameterization " \
              f" --chg_method {chg_method} " \
              f" --percent_par {percent_par} " \
              f" --optimization_method {optimization_method} " \
              f" --sdf_file {basename(sdf_file)} " \
              f" --ats_types_pattern {ats_types_pattern} " \
              f" --ref_chgs_file {basename(ref_chgs_file)} " \
              f" --data_dir {data_dir} " \
              f" --cpu {cpu} " \
              f" --git_hash {git.Repo(search_parent_directories=True).head.object.hexsha} " \
              f" --num_of_samples {num_of_samples} " \
              f" --subset {subset} " \
              f" --num_of_candidates {num_of_candidates} " \
              f" --random_seed {random_seed} "

    if params_file:
        command += f"--params_file {basename(params_file)} " \

    print("Submitting job in planning system...")
    system(f"ssh dargen3@nympha.metacentrum.cz "
           f"\"export PBS_SERVER=cerit-pbs.cerit-sc.cz; "
           f"cd /storage/praha1/home/dargen3/mach/para_submit; "
           f"./para_submit.sh '{command}' {data_dir} -l select=1:ncpus={cpu}:mem={RAM}gb:scratch_local={RAM}gb:cluster=zenon  -l walltime={walltime}:00:00 ;" 
           f"cd .. ; cp -r mach.py modules/ {data_dir}/source_code/ ; rm -r {data_dir}/source_code/modules/__pycache__\"")
    print(colored("ok\n", "green"))
