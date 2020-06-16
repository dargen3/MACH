from os import path, makedirs
from shutil import rmtree, copy
from sys import exit

from termcolor import colored


def control_and_copy_input_files(data_dir: str,
                                 files: tuple,
                                 rewriting_with_force: bool):

    print(f'Control presence of output directory {data_dir}...')
    if path.isdir(data_dir):
        if rewriting_with_force:
            rmtree(data_dir)
            print(f"    {data_dir} directory was deleted...")
        else:
            exit(colored(f"ERROR! {data_dir} directory is present. If you want to rewrite it"
                         f"run MACH with -f or --rewriting_with_force option.\n", "red"))
    print(colored("ok\n", "green"))

    print('Control presence of input files... \n    {}'.format("\n    ".join([file for file in files])))
    for file in files:
        if not path.isfile(file):
            exit(colored(f"ERROR! There is no file {file}.\n", "red"))
    print(colored("ok\n", "green"))

    print(f'Creation of output directory {data_dir} and copying input files...')
    input_files_data_dir = path.join(data_dir, "input_files")
    makedirs(input_files_data_dir)
    makedirs(path.join(data_dir, "output_files"))
    for file in files:
        copy(file, input_files_data_dir)
    print(colored("ok\n", "green"))
