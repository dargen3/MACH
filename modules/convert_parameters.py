from json import load


def _convert_atom_schindler(atom):
    if atom[1] == "hbo":
        return "{}~{}".format(atom[0], atom[2])
    elif atom[1] == "plain":
        return atom[0]
    elif atom[1] == "hbob":
        return "/".join([atom[0], atom[2]])
    elif atom[1] == "hbobhbo":
        return "/~".join([atom[0], atom[2]])


def _convert_atom_racek(atom):
        if "/~" in atom:
            s_atom = atom.split("/")
            return [s_atom[0], "hbobhbo", s_atom[1]]
        elif "/" in atom:
            s_atom = atom.split("/")
            return [s_atom[0], "hbob", s_atom[1]]
        s_atom = atom.split("~")
        if len(s_atom) == 2:
            return [s_atom[0], "hbo", s_atom[1]]
        elif len(s_atom) == 1:
            return [atom, "plain", "*"]


def _convert_bond_schindler(bond):
    bond_atoms = "-".join(*sorted([bond[:2]]))
    if bond[2] in ["hbo", "hbob", "hbobhbo"]:
        return "{}-{}".format(bond_atoms, bond[3])
    elif bond[2] == "plain":
        return "{}".format(bond_atoms)


def _convert_bond_racek(bond):
    s_bond = bond.split("-")
    if "/" in bond:
        return [s_bond[0], s_bond[1], "hbob", bond[-1]]
    elif "/~" in bond:
        return [s_bond[0], s_bond[1], "hbobhbo", bond[-1]]
    elif "~" in bond:
        return [s_bond[0], s_bond[1], "hbo", bond[-1]]
    else:
        return [s_bond[0], s_bond[1], "plain", "*"]


def convert_parameters_schindler(parameters):
    atoms_data = list(parameters["atom"]["data"])
    parameters["atom"]["data"] = {}
    for atom in atoms_data:
        parameters["atom"]["data"][_convert_atom_schindler(atom["key"])] = atom["value"]

    if "bond" in parameters:
        bonds_data = list(parameters["bond"]["data"])
        parameters["bond"]["data"] = {}
        for bond in bonds_data:
            parameters["bond"]["data"][_convert_bond_schindler(bond["key"])] = bond["value"][0]

    if "common" in parameters:
        common_data = dict(parameters["common"])
        parameters["common"] = {}
        for parameter_name, value in zip(common_data["names"], common_data["values"]):
            parameters["common"][parameter_name] = value

    parameters["metadata"]["atomic_types_pattern"] = atoms_data[0]["key"][1]

    return parameters
