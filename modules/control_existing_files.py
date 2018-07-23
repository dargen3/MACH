from os import path
from sys import exit
from termcolor import colored

def control_existing_files(files_data, rewriting_with_force=False):
    print("Control of the presence of the files {}... ".format(", ".join([file[0] for file in files_data])))
    exit_status = False
    for file, presence in files_data:
        if not presence and rewriting_with_force:
            continue
        if path.isfile(file) != presence:
            exit_status = True
            if presence:
                print(colored("There is no \"{}\" file.".format(file), "red"))
            else:
                print(colored("File {} is preset. If you want to rewrite file, "
                              "run MACH with -f or --rewriting_with_force options.".format(file), "red"))
    if exit_status:
        exit("")
    print(colored("ok\n", "green"))