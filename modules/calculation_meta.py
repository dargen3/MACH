from os.path import basename
from os import system


def calculation_meta(sdf, method, parameters, charges, atomic_types_pattern, RAM, walltime):
    command = "./mach.py --mode calculation --parameters {} --method {} --sdf {} --charges {}  -f --atomic_types_pattern {}"\
        .format(basename(parameters), method, basename(sdf), charges, atomic_types_pattern)
    system("./modules/calculation_meta.sh {} {} {} '{}' {} {}".format(parameters, sdf, charges, command, RAM, walltime))
