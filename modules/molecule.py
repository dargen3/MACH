from numpy import array, float32, int64, int16, zeros, float64, concatenate
from scipy import spatial
from sys import exit
from termcolor import colored
from .atom import Atom
from .bond import Bond
from numba import jit


class MoleculeChg:
    def __init__(self, charges, name, atomic_types=None):
        self.name = name
        self.charges = array(charges, dtype=float32)
        if atomic_types:
            self.atomic_types = sorted(set(atomic_types))

    def atoms_representation(self, _):
        return self.atomic_types


def create_atom_high_bond(num_of_atoms, bonds, atomic_symbols):
    highest_bonds = [1] * num_of_atoms
    for (atom1, atom2), bond_type in bonds:
        if highest_bonds[atom1] < bond_type:
            highest_bonds[atom1] = bond_type
        if highest_bonds[atom2] < bond_type:
            highest_bonds[atom2] = bond_type
    return ["{}~{}".format(a, b) for a, b in zip(atomic_symbols, highest_bonds)]


class Molecule:
    def __init__(self, name, num_of_atoms, atomic_symbols, atomic_coordinates, bonds):
        self.name = name
        self.num_of_atoms = num_of_atoms
        self.atoms = []
        atoms_high_bonds = create_atom_high_bond(self.num_of_atoms, bonds, atomic_symbols)
        for index, (cor, atomic_symbol, symbol_high_bond) in enumerate(zip(atomic_coordinates, atomic_symbols, atoms_high_bonds)):
            self.atoms.append(Atom(cor, atomic_symbol, symbol_high_bond, index))
        self.bonds = []
        for (a1, a2), type in bonds:
            self.bonds.append(Bond(self.atoms[a1], self.atoms[a2], type))

    @property
    def atomic_coordinates(self):
        return array([atom.coordinates for atom in self.atoms], dtype=float32)

    def bonds_representation(self, representation):
        return [bond.get_representation(representation) for bond in self.bonds]

    def atoms_representation(self, pattern):
        return [getattr(atom, pattern) for atom in self.atoms]

    def distances(self):
        return array([value for index, line in enumerate(spatial.distance.cdist(self.atomic_coordinates, self.atomic_coordinates)) for value in line[index + 1:]], dtype=float32)

    def symbolic_numbers_atoms(self, method):
        return array([method.atomic_types.index(atomic_type) for atomic_type in self.atoms_representation(method.atomic_types_pattern)], dtype=int16)

    def symbolic_numbers_bonds(self, method):
        return array([method.bond_types.index("bond-{}".format(bond)) for bond in self.bonds_representation("{}_{}".format(method.atomic_types_pattern, method.atomic_types_pattern))], dtype=int16)

    def bonds_without_bond_type(self):
        return concatenate(self.bonds_representation("index_index"))

    def num_of_bonds_mul_two(self):
        return [len(self.bonds) * 2]

    def num_of_bonds(self):
        return [len(self.bonds)]

    def MGC_matrix(self):
        matrix = zeros((self.num_of_atoms, self.num_of_atoms), dtype=float64)
        for x in range(self.num_of_atoms):
            matrix[x][x] = 1
        for atom1, atom2, bond_type in self.bonds_representation("index_index_type"):
            matrix[atom1][atom1] += bond_type
            matrix[atom2][atom2] += bond_type
            matrix[atom1][atom2] -= bond_type
            matrix[atom2][atom1] -= bond_type
        return array([value for index, line in enumerate(matrix) for value in line[index:]], dtype=float64)

    def DENR_matrix(self):
        matrix = zeros((self.num_of_atoms, self.num_of_atoms), dtype=float64)
        for atom1, atom2, bond_type in self.bonds_representation("index_index_type"):
            matrix[atom1][atom1] += bond_type
            matrix[atom2][atom2] += bond_type
            matrix[atom1][atom2] -= bond_type
            matrix[atom2][atom1] -= bond_type
        return array([value for index, line in enumerate(matrix) for value in line[index:]], dtype=float64)


    def __len__(self):
        return self.num_of_atoms
