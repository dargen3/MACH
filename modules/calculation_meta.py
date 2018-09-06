from os.path import basename
from os import system


def calculation_meta(sdf, method, parameters, charges, RAM, walltime):  # only for my usage
    command = "./mach.py --mode calculation --parameters {} --method {} --sdf {} --charges {}  -f"\
        .format(basename(parameters), method, basename(sdf), charges)
    system("./modules/calculation_meta.sh {} {} {} '{}' {} {}".format(parameters, sdf, charges, command, RAM, walltime))

