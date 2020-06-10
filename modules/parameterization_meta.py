from os import system
from os.path import basename

import git
from termcolor import colored


def parameterization_meta(sdf_file: str,
                          ref_chgs_file: str,
                          chg_method: str,
                          params_file: str,
                          optimization_method: str,
                          minimization_method: str,
                          num_of_samples: int,
                          num_of_candidates: int,
                          par_subset: int,
                          cpu: int,
                          RAM: int,
                          walltime: int,
                          random_seed: int,
                          data_dir: str):

    print("Copying of data to MetaCentrum...")
    system(f"ssh dargen3@tarkil.metacentrum.cz "
           f"'cd /storage/praha1/home/dargen3/mach; "
           f"mkdir {data_dir}; "
           f"mkdir {data_dir}/input_files; "
           f"mkdir {data_dir}/output_files'")
    system(f"scp {sdf_file} {ref_chgs_file} {params_file} dargen3@tarkil.metacentrum.cz:/storage/praha1/home/dargen3/mach/{data_dir}/input_files")
    print(colored("ok\n", "green"))

    params_file = basename(params_file)
    sdf_file = basename(sdf_file)
    ref_chgs_file = basename(ref_chgs_file)

    command = f"/storage/praha1/home/dargen3/miniconda3/bin/python3.7 mach.py " \
              f"--mode parameterization " \
              f"--chg_method {chg_method} " \
              f"--optimization_method {optimization_method} " \
              f"--minimization_method {minimization_method} " \
              f"--params_file {params_file} " \
              f"--sdf_file {sdf_file} " \
              f"--ref_chgs_file {ref_chgs_file} " \
              f"--data_dir {data_dir} " \
              f"--cpu {cpu} " \
              f"--git_hash {git.Repo(search_parent_directories=True).head.object.hexsha} " \
              f"--num_of_samples {num_of_samples} " \
              f"--par_subset {par_subset} " \
              f"--num_of_candidates {num_of_candidates} " \
              f"--random_seed {random_seed}"

    print("Submitting job in planning system...")
    system(f"""ssh dargen3@nympha.metacentrum.cz "export PBS_SERVER=cerit-pbs.cerit-sc.cz; cd /storage/praha1/home/dargen3/mach/para_submit; ./para_submit.sh '{command}' {data_dir} -l select=1:ncpus={cpu}:mem={RAM}gb:scratch_local={RAM}gb -l walltime={walltime}:00:00" """)
    print(colored("ok\n", "green"))
