class Bond:
    def __init__(self, atom1, atom2, type_of_bond):
        self.atom1 = atom1
        self.atom2 = atom2
        self.type_of_bond = type_of_bond

    def get_reprezentation(self, reprezentation):
        if reprezentation == "index_index":
            return self.atom1.index, self.atom2.index
        elif reprezentation == "index_index_type":
            return self.atom1.index, self.atom2.index, self.type_of_bond
        elif reprezentation == "atomic_symbol_high_bond_atomic_symbol_high_bond":
            return "-".join(sorted([self.atom1.atomic_symbol_high_bond, self.atom2.atomic_symbol_high_bond]))
        elif reprezentation == "atomic_symbol_atomic_symbol":
            return "-".join(sorted([self.atom1.atomic_symbol, self.atom2.atomic_symbol]))
