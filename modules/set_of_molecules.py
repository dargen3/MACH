from .molecule import Molecule, MoleculeChg
from .control_order_of_molecules import control_order_of_molecules
from sys import exit
from termcolor import colored
from collections import Counter
from tabulate import tabulate
from numpy import array, float32


def sort(a, b):
    if a > b:
        return b, a
    return a, b

class ArciSet:
    def __init__(self, file):
        self.molecules = []
        self.file = file

    def __len__(self):
        return self.num_of_molecules

    def __getitem__(self, index):
        return self.molecules[index]

    def __iter__(self):
        return iter(self.molecules)


class SetOfMolecules(ArciSet):
    def __init__(self, file, num_of_molecules=None):
        print("Loading of set of molecules from {}...".format(file))
        super().__init__(file)
        with open(file, "r") as sdf:
            molecules_data = sdf.read()
        if molecules_data[-5:].strip() != "$$$$":
            exit(colored("{} is not valid sdf file. Last line is not $$$$.\n".format(sdf.name), "red"))
        molecules_data = [x.splitlines() for x in molecules_data.split("$$$$\n")]
        num_of_all_molecules = len(molecules_data) - 1
        if not num_of_molecules:
            self.num_of_molecules = num_of_all_molecules
        else:
            if num_of_molecules > num_of_all_molecules:
                exit(colored(("ERROR! Number of molecules is only {}!".format(num_of_all_molecules)), "red"))
            self.num_of_molecules = num_of_molecules
        for molecule_data in molecules_data[:self.num_of_molecules]:
            type_of_sdf_record = molecule_data[3][-5:]
            if type_of_sdf_record == "V2000":
                self.load_sdf_v2000(molecule_data)
            elif type_of_sdf_record == "V3000":
                self.load_sdf_v3000(molecule_data)
            else:
                exit(colored("{} if not valid sdf file.\n".format(sdf), "red"))
        self.num_of_atoms = sum([len(molecule) for molecule in self.molecules])
        print(colored("ok\n", "green"))

    def load_sdf_v2000(self, molecular_data):
        name = molecular_data[0]
        info_line = molecular_data[3]
        num_of_atoms = int(info_line[:3])
        num_of_bonds = int(info_line[3:6])
        atomic_symbols, atomic_coordinates, bonds = [], [], []
        for atom_line in molecular_data[4: num_of_atoms + 4]:
            line = atom_line.split()
            atomic_coordinates.append((float(line[0]), float(line[1]), float(line[2])))
            atomic_symbols.append(line[3])
        for bond_line in molecular_data[num_of_atoms + 4: num_of_atoms + num_of_bonds + 4]:
            bonds.append((sort(int(bond_line[:3]), int(bond_line[3:6])), int(bond_line[8])))
        self.molecules.append(Molecule(name, num_of_atoms, atomic_symbols, atomic_coordinates, bonds))

    def load_sdf_v3000(self, molecular_data):
        name = molecular_data[0]
        info_line = molecular_data[5].split()
        num_of_atoms = int(info_line[3])
        num_of_bonds = int(info_line[4])
        atomic_symbols, atomic_coordinates, bonds = [], [], []
        for atom_line in molecular_data[7: num_of_atoms + 7]:
            line = atom_line.split()
            atomic_coordinates.append((float(line[4]), float(line[5]), float(line[6])))
            atomic_symbols.append(line[3])
        for bond_line in molecular_data[num_of_atoms + 9: num_of_atoms + num_of_bonds + 9]:
            line = bond_line.split()
            bonds.append((sort(int(line[4]), int(line[5])), int(line[3])))
        self.molecules.append(Molecule(name, num_of_atoms, atomic_symbols, atomic_coordinates, bonds))

    def info(self, atomic_types_pattern, file=None):  # file only for my usage
        counter = Counter()
        if atomic_types_pattern == "atom":
            for molecule in self.molecules:
                counter.update(molecule.atomic_symbols)
        elif atomic_types_pattern == "atom_high_bond":
            for molecule in self.molecules:
                counter.update(molecule.atomic_symbols_high_bond)
        table = []
        for atom, count in sorted(counter.items()):
            table.append((atom, count, round(count / (self.num_of_atoms / 100), 2)))
        data = """Statistics data from set of molecules from {}
Number of molecules:   {}
Number of atoms:       {}
Number of atoms type:  {}\n
{}\n""".format(self.file, self.num_of_molecules, self.num_of_atoms, len(counter),
           tabulate(table, headers=["Type", "Number", "%"]))
        if file: # only for my usage
            with open(file, "w") as data_file:
                data_file.write(data)
        else:
            print(data)

    def add_ref_charges(self, file, num_of_atomic_types, all_symbolic_numbers):
        with open(file, "r") as charges_file:
            names = [data.splitlines()[0] for data in charges_file.read().split("\n\n")[:-1]][:self.num_of_molecules]
            control_order_of_molecules(names, [molecule.name for molecule in self.molecules], file, self.file)
        print("Loading charges from {}...".format(file))
        with open(file, "r") as charges_file:
            charges = []
            for molecule_data, molecule in zip(charges_file.read().split("\n\n")[:-1], self.molecules):
                molecule_charges = []
                for line in molecule_data.splitlines():
                    try:
                        molecule_charges.append(float(line.split()[2]))
                    except IndexError:
                        pass
                charges.extend(molecule_charges)
                molecule.charges = array(molecule_charges)
        self.ref_charges = array(charges, dtype=float32)
        atomic_types_charges = [[] for _ in range(num_of_atomic_types)]
        for charge, symbolic_number in zip(self.ref_charges, all_symbolic_numbers):
            atomic_types_charges[symbolic_number].append(charge)
        self.ref_atomic_types_charges = array([array(chg, dtype=float32) for chg in atomic_types_charges])
        print(colored("ok\n", "green"))


class SetOfMoleculesFromChargesFile(ArciSet):
    def __init__(self, file, ref=True):
        print("Loading of set of molecules from {}...".format(file))
        super().__init__(file)
        with open(self.file, "r") as charges_file:
            molecules_data = [[line.split() for line in molecule.splitlines()]
                              for molecule in charges_file.read().split("\n\n")[:-1]]
            self.num_of_molecules = len(molecules_data)
            self.names = []
            all_charges = []
            atomic_types_charges = {}
            for molecule in molecules_data:
                self.names.append(molecule[0][0])
                atomic_symbols = [atom_line[1] for atom_line in molecule[2:]]
                charges = array([float(atom_line[2]) for atom_line in molecule[2:]])
                self.molecules.append(MoleculeChg(charges))
                for atomic_symbol, charge in zip(atomic_symbols, charges):
                    atomic_types_charges.setdefault(atomic_symbol, []).append(charge)
                all_charges.extend(charges)
            for atomic_symbol, charge in atomic_types_charges.items():
                atomic_types_charges[atomic_symbol] = array(charge)
            if ref:
                self.ref_charges = array(all_charges, dtype=float32)
            else:
                self.all_charges = array(all_charges, dtype=float32)
            self.atomic_types_charges = atomic_types_charges
        print(colored("ok\n", "green"))
