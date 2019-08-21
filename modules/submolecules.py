from .set_of_molecules import SetOfMolecules
from .control_existing import control_existing_files
from scipy.spatial import cKDTree
from os.path import basename
from collections import defaultdict
from operator import itemgetter
from termcolor import colored


def create_submolecules(sdf, number_of_molecules=None, cutoff=8):
    set_of_molecules = SetOfMolecules(sdf, num_of_molecules=number_of_molecules)
    print("Creating of submolecules...")
    new_sdf = basename(sdf) + ".submolecules"
    sdf_list = []
    number_of_submolecules = 0
    for molecule in set_of_molecules:
        number_of_submolecules += molecule.num_of_atoms
        atomic_coordinates = molecule.atomic_coordinates
        atoms = [" ".join(atom) for atom in [[*map(str, cor), sym] for cor, sym in zip(atomic_coordinates, molecule.atoms_representation("plain"))]]
        bonds = defaultdict(list)
        for a1, a2, bond_type in molecule.bonds_representation("index_index_type"):
            bonds[a1].append((a2, bond_type))
        tree = cKDTree(atomic_coordinates)
        for index, atom_c in enumerate(atomic_coordinates):
            print(index)
            indices = tree.query_ball_point(atom_c, cutoff)
            indices.remove(index)
            indices.insert(0, index)
            indices_set = set(indices)
            new_indices = {i: index for index, i in enumerate(indices, start=1)}
            new_molecule_bonds = []


            for a1 in indices:
                for a2, bond_type in bonds[a1]:
                    if a2 in indices_set:
                        new_molecule_bonds.append("{0:>3}{1:>3}  {2}".format(new_indices[a1], new_indices[a2], bond_type))





            sdf_list.append("""{}


{}  0  0  0  0  0  0  0  0999 V2000
{}
{}
M  END
$$$$\n""".format(f"{molecule.name}~~~{index}",
                 "{0:>3}{1:>3}".format(len(indices), len(new_molecule_bonds)),
                 "\n".join(itemgetter(*indices)(atoms)),
                 "\n".join(new_molecule_bonds)))


    with open(new_sdf, "w") as new_sdf_file:
        new_sdf_file.write("".join(sdf_list))
    print(f"    {number_of_submolecules} submolecules was created.\n" + colored("ok\n", "green"))

    return number_of_submolecules, new_sdf