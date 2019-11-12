from numba import jitclass
from numba.types import string, uint32, float32, uint16, ListType
from numba.typed import List
from numpy import empty, float32 as npfloat32, int32 as npint32


@jitclass({"name": string,
           "num_of_atoms": uint32,
           "atomic_coordinates": float32[:, :],
           "atoms_id": uint16[:],
           "bonds_id": uint16[:],
           "atoms_representation": ListType(string),
           "bonds_representation": ListType(string),
           "distance_matrix": float32[:, :],
           "bonds": uint32[:, :],
           "ref_charges": float32[:],
           "emp_charges": float32[:]})
class Molecule:
    def __init__(self, name, num_of_atoms, atomic_coordinates, atoms_representation, bonds, bonds_representation):
        self.name = name
        self.num_of_atoms = num_of_atoms
        self.atomic_coordinates = atomic_coordinates
        self.atoms_representation = atoms_representation
        self.bonds = bonds
        self.bonds_representation = bonds_representation


def create_molecule_from_charges(name, atoms_representation, ref_charges, emp_charges):
    bonds = List()
    bonds.append("")
    chg_molecule = Molecule(name, len(atoms_representation), empty((0, 0), dtype=npfloat32), atoms_representation, empty((0, 0), dtype=npint32), bonds)
    chg_molecule.ref_charges = ref_charges
    chg_molecule.emp_charges = emp_charges
    return chg_molecule

