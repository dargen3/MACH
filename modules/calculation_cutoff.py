from .control_existing import control_existing_files
from .molecule import Molecule
from .set_of_molecules import create_set_of_molecules, SetOfMolecules, create_method_data
from .input_output import write_charges_to_file
from scipy.spatial import cKDTree
from collections import defaultdict
from operator import itemgetter
from termcolor import colored
from importlib import import_module
from numpy import array, concatenate, array_split, int32, linalg
from numba.typed import List


class CalculationCutoff:
    def __init__(self, sdf, method, parameters, charges, atomic_types_pattern, rewriting_with_force):
        control_existing_files([(sdf, True, "file"),
                                (charges, False, "file")], rewriting_with_force)
        set_of_molecules = create_set_of_molecules(sdf, atomic_types_pattern)
        method = getattr(import_module("modules.methods"), method)()
        method.load_parameters(parameters, set_of_molecules, "calculation", atomic_types_pattern)
        print("Creation of submolecules and calculation of charges... ")
        results = []
        for molecule in set_of_molecules.molecules:
            atomic_coordinates = molecule.atomic_coordinates
            bonds = defaultdict(list)
            for a1, a2, bond_type in molecule.bonds:
                bonds[a1].append((a2, bond_type))
            tree = cKDTree(atomic_coordinates)
            index = 0
            for atomic_coordinates_part in array_split(atomic_coordinates, len(atomic_coordinates)//2000+1):
                new_molecules = List()
                num_of_atoms_in_set = 0
                for atom_c in atomic_coordinates_part:
                    print(index)
                    indices = tree.query_ball_point(atom_c, 12)
                    num_of_atoms_in_set += len(indices)
                    indices.remove(index)
                    indices.insert(0, index)
                    indices_set = set(indices)
                    new_indices = {i: index-1 for index, i in enumerate(indices, start=1)}
                    new_molecule_bonds = List()
                    new_molecule_bonds_representation = List()
                    new_atoms_representation = List()
                    [new_atoms_representation.append(atom_reprezentation) for atom_reprezentation in itemgetter(*indices)(molecule.atoms_representation)]
                    for a1 in indices:
                        for a2, bond_type in bonds[a1]:
                            if a2 in indices_set:
                                new_indice_a1 = new_indices[a1]
                                new_indice_a2 = new_indices[a2]
                                new_molecule_bonds.append(array((new_indice_a1, new_indice_a2, bond_type), dtype=int32))
                                new_molecule_bonds_representation.append("{}-{}".format("-".join(sorted([new_atoms_representation[new_indice_a1], new_atoms_representation[new_indice_a2]])), bond_type))

                    new_molecules.append(Molecule("None",
                                                  len(indices),
                                                  array(itemgetter(*indices)(atomic_coordinates)),
                                                  new_atoms_representation,
                                                  array(new_molecule_bonds),
                                                  new_molecule_bonds_representation))


                    index += 1
                new_set_of_molecules = SetOfMolecules(new_molecules, "None", len(new_molecules), num_of_atoms_in_set)
                create_method_data(method, new_set_of_molecules)
                try:
                    method.calculate(new_set_of_molecules)
                except (linalg.linalg.LinAlgError, ZeroDivisionError) as e:
                    print(e)
                indicessss = []
                indexx = 0
                for num_of_atom in [molecule.num_of_atoms for molecule in new_set_of_molecules.molecules]:
                    indicessss.append(indexx)
                    indexx += num_of_atom
                indicessss = array(indicessss)
                results.append(method.results[indicessss])
        print(colored("ok\n", "green"))
        write_charges_to_file(charges, concatenate(results), set_of_molecules)

