from numba.typed import List
from numpy import array, int32, float32
from termcolor import colored

from .control_order_of_molecules import control_order_of_molecules
from .molecule import Molecule


def add_charges_to_set_of_molecules(set_of_molecules, ref_chg_file):
    with open(ref_chg_file, "r") as reference_charges_file:
        names = [data.splitlines()[0] for data in reference_charges_file.read().split("\n\n")[:-1]][:set_of_molecules.num_of_molecules]
        print(f"Loading charges from {ref_chg_file}...")
        control_order_of_molecules(names, [molecule.name for molecule in set_of_molecules.molecules], ref_chg_file, set_of_molecules.sdf_file)
        reference_charges_file.seek(0)
        for molecule_data, molecule in zip(reference_charges_file.read().split("\n\n")[:-1], set_of_molecules.molecules):
            molecule_charges = []
            for line in molecule_data.splitlines()[2:]:
                molecule_charges.append(float(line.split()[2]))
            molecule.ref_charges = array(molecule_charges, dtype=float32)
    print(colored("ok\n", "green"))


def write_charges_to_file(charges, results, set_of_molecules):
    print("Writing charges to {}...".format(charges))
    with open(charges, "w") as file_with_results:
        count = 0
        for molecule in set_of_molecules.molecules:
            file_with_results.write("{}\n{}\n".format(molecule.name, molecule.num_of_atoms))
            for index, atom in enumerate(molecule.atoms_representation):
                file_with_results.write("{0:>3} {1:>3} {2:>15}\n".format(index + 1, atom.split("~")[0], str(float("{0:.6f}".format(results[count])))))
                count += 1
            file_with_results.write("\n")
    print(colored("ok\n", "green"))


def _sort(a, b):
    if a > b:
        return b - 1, a - 1
    return a - 1, b - 1


def _create_atoms_bonds_representation(num_of_atoms, atomic_symbols, bonds, atomic_types_pattern):
    def _create_atom_highest_bond(num_of_atoms, bonds, atomic_symbols):
        highest_bonds = [1] * num_of_atoms
        for ba1, ba2, type in bonds:
            if highest_bonds[ba1] < type:
                highest_bonds[ba1] = type
            if highest_bonds[ba2] < type:
                highest_bonds[ba2] = type
        return [atomic_symbol + "~" + str(hbo) for atomic_symbol, hbo in zip(atomic_symbols, highest_bonds)]

    # create atoms_representation (one great string)
    # plain (plain atom) = "C;C;H;N;..."
    # hbo (highest bond order) = "C~1;C~2;H~1;N~3;..."
    # hbob (highest bond order and bonded atoms) = "C~1/CHHH;C~2/CHH;H~1/C;N~3/HHC;..."
    atoms_representation = List()
    if atomic_types_pattern == "plain":
        [atoms_representation.append(atom) for atom in atomic_symbols]
    elif atomic_types_pattern == "hbo":
        [atoms_representation.append(atom) for atom in _create_atom_highest_bond(num_of_atoms, bonds, atomic_symbols)]
    elif atomic_types_pattern in ["hbob", "hbob_sb"]:
        bonded_atoms = [[] for _ in range(num_of_atoms)]
        for ba1, ba2, _ in bonds:
            bonded_atoms[ba1].append(atomic_symbols[ba2])
            bonded_atoms[ba2].append(atomic_symbols[ba1])
        [atoms_representation.append("{}/{}".format(hbo, "".join(sorted(bonded_atoms)))) for hbo, bonded_atoms in zip(_create_atom_highest_bond(num_of_atoms, bonds, atomic_symbols), bonded_atoms)]
    elif atomic_types_pattern in ["hbobhbo"]:
        bonded_atoms = [[] for _ in range(num_of_atoms)]
        hbo = _create_atom_highest_bond(num_of_atoms, bonds, atomic_symbols)
        hbo_without_spec = [atom.replace("~", "") for atom in hbo]

        for ba1, ba2, _ in bonds:
            bonded_atoms[ba1].append(hbo_without_spec[ba2])
            bonded_atoms[ba2].append(hbo_without_spec[ba1])

        [atoms_representation.append("{}/{}".format(hbo, "".join(sorted(bonded_atoms)))) for hbo, bonded_atoms in zip(hbo, bonded_atoms)]

    # create bonds_representation (one great string)
    # plain (plain atom) = "C_H;C_N;..."
    # hbo (highest bond order) = "C~1_H~1;C~2_N~2;..."
    # hbob (highest bond order and bonded atoms) = "C~1/HHH_H~1/C;C~2/CHH_N~2/CH;..."
    bonds_representation = List()
    if atomic_types_pattern in ["hbo", "hbob", "hbobhbo"]:
        [bonds_representation.append("{}-{}".format("-".join(sorted([atoms_representation[ba1], atoms_representation[ba2]])), type)) for ba1, ba2, type in bonds]
    elif atomic_types_pattern == "hbob_sb":
        hbo = _create_atom_highest_bond(num_of_atoms, bonds, atomic_symbols)
        [bonds_representation.append("{}-{}".format("-".join(sorted([hbo[ba1], hbo[ba2]])), type)) for ba1, ba2, type in bonds]
    else:
        [bonds_representation.append("-".join(sorted([atoms_representation[ba1], atoms_representation[ba2]]))) for ba1, ba2, _ in bonds]
    return atoms_representation, bonds_representation


def load_sdf_v2000(molecular_data, atomic_types_pattern):
    name = molecular_data[0]
    info_line = molecular_data[3]
    num_of_atoms = int(info_line[:3])
    num_of_bonds = int(info_line[3:6])
    atomic_symbols, atomic_coordinates, bonds = [], [], []

    # read atoms lines
    for atom_line in molecular_data[4: num_of_atoms + 4]:
        c1, c2, c3, symbol = atom_line.split()[:4]
        atomic_coordinates.append([float(c1), float(c2), float(c3)])
        atomic_symbols.append(symbol)

    # read bond lines
    for bond_line in molecular_data[num_of_atoms + 4: num_of_atoms + num_of_bonds + 4]:
        a1, a2 = _sort(int(bond_line[:3]), int(bond_line[3:6]))
        bonds.append((a1, a2, int(bond_line[8])))

    atoms_representation, bonds_representation = _create_atoms_bonds_representation(num_of_atoms, atomic_symbols, bonds, atomic_types_pattern)

    return Molecule(name, num_of_atoms, array(atomic_coordinates, dtype=float32), atoms_representation, array(bonds, dtype=int32), bonds_representation)


def load_sdf_v3000(molecular_data, atomic_types_pattern):
    name = molecular_data[0]
    info_line = molecular_data[5].split()
    num_of_atoms = int(info_line[3])
    num_of_bonds = int(info_line[4])
    atomic_symbols, atomic_coordinates, bonds = [], [], []

    # read atoms lines
    for atom_line in molecular_data[7: num_of_atoms + 7]:
        line = atom_line.split()
        atomic_coordinates.append(array([float(line[4]), float(line[5]), float(line[6])], dtype=float32))
        atomic_symbols.append(line[3])

    # read bond lines
    for bond_line in molecular_data[num_of_atoms + 9: num_of_atoms + num_of_bonds + 9]:
        line = bond_line.split()
        a1, a2 = _sort(int(line[4]), int(line[5]))
        bonds.append((a1, a2, int(line[3])))

    atoms_representation, bonds_representation = _create_atoms_bonds_representation(num_of_atoms, atomic_symbols, bonds, atomic_types_pattern)

    return Molecule(name, num_of_atoms, array(atomic_coordinates, dtype=float32), atoms_representation, array(bonds, dtype=int32), bonds_representation)
