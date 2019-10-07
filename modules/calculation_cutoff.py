from .control_existing import control_existing_files
from .molecule import Molecule
from .set_of_molecules import SetOfMolecules
from .output_files import write_charges_to_file
from scipy.spatial import cKDTree
from collections import defaultdict
from operator import itemgetter
from termcolor import colored
from importlib import import_module
from numpy import array, concatenate, array_split
from .bond import Bond

class CalculationCutoff:
    def __init__(self, sdf, method, parameters, charges, atomic_types_pattern, rewriting_with_force):
        control_existing_files([(sdf, True, "file"),
                                (charges, False, "file")], rewriting_with_force)
        set_of_molecules = SetOfMolecules(sdf)
        method = getattr(import_module("modules.methods"), method)()
        method.load_parameters(parameters, set_of_molecules, "calculation", atomic_types_pattern)
        print("Creation of submolecules and calculation of charges... ")
        results = []
        for molecule in set_of_molecules:
            atomic_coordinates = molecule.atomic_coordinates
            atoms = molecule.atoms
            bonds = defaultdict(list)
            for a1, a2, bond_type in molecule.bonds_representation("index_index_type"):
                bonds[a1].append((a2, bond_type))
            tree = cKDTree(atomic_coordinates)
            index = 0
            for atomic_coordinates_part in array_split(atomic_coordinates, len(atomic_coordinates)//2000+1):
                new_molecules = []
                for atom_c in atomic_coordinates_part:
                    print(index)
                    indices = tree.query_ball_point(atom_c, 4)
                    indices.remove(index)
                    indices.insert(0, index)
                    indices_set = set(indices)
                    new_indices = {i: index-1 for index, i in enumerate(indices, start=1)}
                    new_molecules_bonds = []
                    for a1 in indices:
                        for a2, bond_type in bonds[a1]:
                            if a2 in indices_set:
                                new_molecules_bonds.append(Bond(new_indices[a1], new_indices[a2], bond_type))
                    new_molecules.append(Molecule(None, itemgetter(*indices)(atoms), new_molecules_bonds))
                    index += 1
                new_set_of_molecules = SetOfMolecules(new_molecules, molecules_list=True)
                new_set_of_molecules.create_method_data(method)
                try:
                    method.calculate(new_set_of_molecules)
                except (linalg.linalg.LinAlgError, ZeroDivisionError) as e:
                    print(e)
                indicessss = []
                indexx = 0
                for num_of_atom in [len(molecule) for molecule in new_set_of_molecules]:
                    indicessss.append(indexx)
                    indexx += num_of_atom
                indicessss = array(indicessss)
                results.append(method.results[indicessss])
        print(colored("ok\n", "green"))
        write_charges_to_file(charges, concatenate(results), set_of_molecules)

