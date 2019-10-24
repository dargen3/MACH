# from numba import jitclass
# from numba.types import string, uint8, float32
#
# spec = [("coordinates", float32[:]), ("plain", string), ("index", uint8)]
#
# @jitclass(spec)
class Atom:
    def __init__(self, coordinates, atomic_symbol, index):
        self.coordinates = coordinates
        self.plain = atomic_symbol
        self.index = index

    def get_representation(self, pattern):
        if pattern == "plain":
            return self.atomic_symbol
        elif pattern == "hbo":
            return self.atomic_symbol_high_bond
        elif pattern == "hbob":
            return self.hbob
