from numba.experimental import jitclass
from numba.core.types import string, uint32, float32, uint16, int16, ListType
import numpy as np

@jitclass({"name": string,
           "num_of_ats": uint32,
           "num_of_bonds": uint32,
           "ats_coordinates": float32[:, :],
           "ats_ids": uint16[:],
           "bonds_ids": uint16[:],
           "ats_srepr": ListType(string),
           "bonds_srepr": ListType(string),
           "distance_matrix": float32[:, :],
           "bonds": uint32[:, :],
           "ref_chgs": float32[:],
           "emp_chgs": float32[:]})
class Molecule:
    def __init__(self, name, num_of_ats, num_of_bonds, ats_coordinates, ats_srepr, bonds, bonds_srepr):
        self.name = name
        self.num_of_ats = num_of_ats
        self.num_of_bonds = num_of_bonds
        self.ats_coordinates = ats_coordinates
        self.ats_srepr = ats_srepr
        self.bonds = bonds
        self.bonds_srepr = bonds_srepr

    @property
    def total_chg(self):
        return round(np.sum(self.ref_chgs))