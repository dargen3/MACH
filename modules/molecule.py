from numpy import array, float32, int64, int16
from scipy import spatial
from sys import exit
from termcolor import colored


class MoleculeChg:
    def __init__(self, charges):
        self.charges = array(charges, dtype=float32)


def create_atom_high_bond(num_of_atoms, bonds, atomic_symbols):
    highest_bonds = [1] * num_of_atoms
    for bond in bonds:
        bond_type = bond[1]
        atoms = bond[0]
        atom1 = atoms[0] - 1
        atom2 = atoms[1] - 1
        if highest_bonds[atom1] < bond_type:
            highest_bonds[atom1] = bond_type
        if highest_bonds[atom2] < bond_type:
            highest_bonds[atom2] = bond_type
    return ["{}~{}".format(a, b) for a, b in zip(atomic_symbols, highest_bonds)]


class Molecule:
    def __init__(self, name, num_of_atoms, atomic_symbols, atomic_coordinates, bonds):
        self.name = name
        self.num_of_atoms = num_of_atoms
        self.atomic_symbols = atomic_symbols
        self.atomic_coordinates = array(atomic_coordinates)
        self.bonds = bonds
        self.atomic_symbols_high_bond = create_atom_high_bond(self.num_of_atoms, self.bonds, self.atomic_symbols)

    def distances(self):
        return array([value for index, line in enumerate(spatial.distance.cdist(self.atomic_coordinates, self.atomic_coordinates)) for value in line[index + 1:]], dtype=float32)

    def symbolic_numbers(self, method):
        try:
            if method.atomic_types_pattern == "atom":
                return array([method.atomic_types.index(atomic_type) for atomic_type in self.atomic_symbols], dtype=int16)
            elif method.atomic_types_pattern == "atom_high_bond":
                return array([method.atomic_types.index(atomic_type) for atomic_type in self.atomic_symbols_high_bond], dtype=int16)
        except ValueError as VE:
            exit(colored("{} atomic type is not defined in parameters.".format(str(VE).split()[0][1:-1]), "red"))

    def bonds_without_bond_type(self):
        return array([atom for bond in self.bonds for atom in bond[0]], dtype=int64) - 1

    def num_of_bonds_mul_two(self):
        return array([len(self.bonds)*2], dtype=int64) # domyslet.. je to nanic

    def __len__(self):
        return self.num_of_atoms
