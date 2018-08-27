from os import path
from sys import exit
from termcolor import colored


def data_existing(object, object_type):
    if object_type == "file":
        return path.isfile(object)
    elif object_type == "directory":
        return path.isdir(object)


def control_existing_files(objects_data, rewriting_with_force=False):
    print("Control presence of files and directories {}... ".format(", ".join([object[0] for object in objects_data])))
    exit_status = False
    for object, presence, object_type in objects_data:
        if not presence and rewriting_with_force:
            continue
        if data_existing(object, object_type) != presence:
            exit_status = True
            if presence:
                print(colored("There is no {} \"{}\".".format(object_type, object), "red"))
            else:
                print(colored("{} {} is present. If you want to rewrite object "
                              "run MACH with -f or --rewriting_with_force option.".format(object_type, object), "red"))
    if exit_status:
        exit("")
    print(colored("ok\n", "green"))

