from numpy import array, float64
from scipy import spatial
from sys import exit
from termcolor import colored


class MoleculeChg:
    def __init__(self, charges):
        self.charges = array(charges, dtype=float64)


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
    def __init__(self, name, num_of_atoms, atomic_symbols, atomic_coordinates, bonds, method=None):
        self.name = name
        self.num_of_atoms = num_of_atoms
        self.atomic_symbols = atomic_symbols
        self.atomic_coordinates = array(atomic_coordinates)
        self.bonds = bonds
        self.atomic_symbols_high_bond = create_atom_high_bond(self.num_of_atoms, self.bonds, self.atomic_symbols)
        if method:
            self.distance_matrix = spatial.distance.cdist(self.atomic_coordinates, self.atomic_coordinates)
            try:
                if method.atomic_types_pattern == "atom":
                    self.symbolic_numbers = array([method.atomic_types.index(atomic_type) for atomic_type in self.atomic_symbols])
                elif method.atomic_types_pattern == "atom_high_bond":
                    self.symbolic_numbers = array([method.atomic_types.index(atomic_type) for atomic_type in self.atomic_symbols_high_bond])
            except ValueError as VE:
                print(str(dir(VE)))
                exit(colored("{}\n".format(str(VE)), "red"))
                # DODELAT!!!!!!!!!! az budeme u netu

    def __len__(self):
        return self.num_of_atoms
