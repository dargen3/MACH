from .set_of_molecules import SetOfMolecules
from .control_existing_files import control_existing_files
from importlib import import_module
from termcolor import colored




class Calculation:
    def __init__(self, sdf, method, parameters, charges, rewriting_with_force):
        self.charges = charges
        control_existing_files(((sdf, True),
                                (parameters, True),
                                (self.charges, False)),
                               rewriting_with_force)
        method = getattr(import_module("modules.methods"), method)()
        method.load_parameters(parameters)
        self.set_of_molecules = SetOfMolecules(sdf, method)
        print("Calculation of charges... ")
        method.load_array_for_results(self.set_of_molecules.num_of_atoms)
        method.calculate(self.set_of_molecules)
        print(colored("ok\n", "green"))
        self.write_to_file(method.results)

    def write_to_file(self, results):
        print("Writing to {}...".format(self.charges))
        with open(self.charges, "w") as file_with_results:
            count = 0
            for molecule in self.set_of_molecules:
                file_with_results.write("{}\n{}\n".format(molecule.name, molecule.num_of_atoms))
                for index, atom in enumerate(molecule.atomic_symbols):
                    file_with_results.write(
                        "{0:>3} {1:>3} {2:>15}\n".format(index + 1, atom, str(float("{0:.6f}".format(results[count])))))
                    count += 1
                file_with_results.write("\n")
        print(colored("ok\n", "green"))
