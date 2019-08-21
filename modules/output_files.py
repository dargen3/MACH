from termcolor import colored


def write_charges_to_file(charges, results, set_of_molecules):
    print("Writing charges to {}...".format(charges))
    with open(charges, "w") as file_with_results:
        count = 0
        for molecule in set_of_molecules:
            file_with_results.write("{}\n{}\n".format(molecule.name, molecule.num_of_atoms))
            for index, atom in enumerate(molecule.atoms_representation("plain")):
                file_with_results.write("{0:>3} {1:>3} {2:>15}\n".format(index + 1, atom, str(float("{0:.6f}".format(results[count])))))
                count += 1
            file_with_results.write("\n")
    print(colored("ok\n", "green"))
