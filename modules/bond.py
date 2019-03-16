class Bond:
    def __init__(self, atom1, atom2, type_of_bond):
        self.atom1 = atom1
        self.atom2 = atom2
        self.type_of_bond = type_of_bond

    def get_representation(self, representation):
        if representation == "index_index":
            return self.atom1.index, self.atom2.index
        elif representation == "index_index_type":
            return self.atom1.index, self.atom2.index, self.type_of_bond
        elif representation == "hbo_hbo":
            return "{}-{}".format("-".join(sorted([self.atom1.hbo, self.atom2.hbo])), self.type_of_bond)
        elif representation == "plain_plain":
            return "-".join(sorted([self.atom1.plain, self.atom2.plain]))
