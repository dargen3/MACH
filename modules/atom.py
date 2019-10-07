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
