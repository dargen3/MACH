from numpy import array, float32, int64, int16, zeros, float64, concatenate
from scipy import spatial
from sys import exit
from termcolor import colored
from numba import jit


class MoleculeChg:
    def __init__(self, charges, name, atomic_types=None):
        self.name = name
        self.charges = array(charges, dtype=float32)
        if atomic_types:
            self.atomic_types = sorted(set(atomic_types))

    def atoms_representation(self, _):
        return self.atomic_types


#hbob
def create_atom_bonded_atoms(atoms, atoms_hbo, bonds_indexes):
    data = []
    for index, atom in enumerate(atoms_hbo):
        bonded_atoms = []
        for i1, i2 in bonds_indexes:
            if index in (i1, i2):
                if index == i1:
                    bonded_atoms.append(atoms[i2])
                else:
                    bonded_atoms.append(atoms[i1])
        data.append("{}/{}".format(atom, "".join(sorted(bonded_atoms))))
    return data







class Molecule:
    def __init__(self, name, atoms, bonds):
        self.name = name
        self.num_of_atoms = len(atoms)
        self.atoms = atoms
        self.bonds = bonds

        #hbob
        for atom, bonded_atoms in zip(self.atoms, create_atom_bonded_atoms(self.atoms_representation("plain"),self.atoms_representation("hbo"), self.bonds_representation("index_index"))):
            atom.hbob = bonded_atoms





    @property
    def atomic_coordinates(self):
        return array([atom.coordinates for atom in self.atoms], dtype=float32)

    def bonds_representation(self, representation):
        return array([bond.get_representation(representation) for bond in self.bonds])

    def atoms_representation(self, pattern):
        return array([getattr(atom, pattern) for atom in self.atoms])

    def distances(self):
        return concatenate([line[index + 1:] for index, line in enumerate(spatial.distance.cdist(self.atomic_coordinates, self.atomic_coordinates))]).astype(float32)

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
