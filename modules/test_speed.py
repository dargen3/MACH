from .set_of_molecules import SetOfMolecules, SubsetOfMolecules
from .control_existing import control_existing_files
from importlib import import_module
from termcolor import colored
from time import time
from datetime import timedelta
from numba import jit

@jit(nopython=True, cache=True)
def lhsclassic(n, samples, high_bond, low_bond):
    random.seed(0)
    cut = linspace(0, 1, samples + 1)
    u = random.rand(samples, n)
    for j in range(n):
        u[random.permutation(arange(samples)), j] = u[:, j] * cut[1] + cut[:samples]
    return u * abs(high_bond - low_bond) + low_bond


class TestSpeed:
    def __init__(self, sdf, method, parameters, subset_heuristic):
        control_existing_files(((sdf, True, "file"),
                                (parameters, True, "file")), False)
        method = getattr(import_module("modules.methods"), method)()
        method.load_parameters(parameters)
        set_of_molecules = SetOfMolecules(sdf)
        if subset_heuristic:
            set_of_molecules = SubsetOfMolecules(set_of_molecules, method)
        set_of_molecules.create_method_data(method)
        print(len(set_of_molecules))
        print("Test of speed. Wait for 5 seconds, please... ")
        method.calculate(set_of_molecules)
        start = time()
        method.calculate(set_of_molecules)
        iterations = int(5 / (time() - start))
        for _ in range(iterations):
            method.calculate(set_of_molecules)
        print(colored("ok\n", "green"))
        one_calculation_time = (time() - start)/iterations
        print("CPU time of one calculation: {} seconds\n\n".format(round(one_calculation_time, 5)))
        print("GM informations:")
        print("Number of parameters: {}".format(len(method.parameters_values)))
        num_of_samples = 1000
        for _ in range(5):
            print("Number of GM samples: {}".format(num_of_samples))
            print("Approximate time of GM without local minimizations: {}"
                  .format(str(timedelta(seconds=int(num_of_samples*one_calculation_time)))))
            samples = lhsclassic(len(method.parameters_values), 1000, -3, 3)
            print("Total size of samples:  {}MB".format(int(samples.nbytes*num_of_samples/1000000000)))
            print()
            num_of_samples *= 10
        print("\n")
