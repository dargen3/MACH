from os import path, makedirs
from shutil import rmtree, copy, copytree
from sys import exit

from termcolor import colored


def control_and_copy_input_files(data_dir: str,
                                 files: tuple):

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
    makedirs(path.join(data_dir, "source_code"))
    source_code_data_dir = path.join(data_dir, "source_code")
    copy("mach.py", source_code_data_dir)
    copytree("modules", path.join(source_code_data_dir, "modules"))
    rmtree(path.join(source_code_data_dir, "modules/__pycache__"))
    print(colored("ok\n", "green"))
