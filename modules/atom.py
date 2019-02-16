class Atom:
    def __init__(self, x, y, z, atomic_symbol, atomic_symbol_high_bond, index):
        self.x = x
        self.y = y
        self.z = z
        self.atomic_symbol = atomic_symbol
        self.atomic_symbol_high_bond = atomic_symbol_high_bond
        self.index = index

    @property
    def coordinates(self):
        return self.x, self.y, self.z

    def get_representation(self, pattern):
        if pattern == "atomic_symbol":
            return self.atomic_symbol
        elif pattern == "atomic_symbol_high_bond":
            return self.atomic_symbol_high_bond
