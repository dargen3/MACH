from .molecule import Molecule, MoleculeChg
from .control_order_of_molecules import control_order_of_molecules
from sys import exit
from termcolor import colored
from collections import Counter
from tabulate import tabulate
from numpy import array, float32, int64, concatenate, random


def sort(a, b):
    if a > b:
        return b-1, a-1
    return a-1, b-1


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
    def __init__(self, file, num_of_molecules_from=None, num_of_molecules_to=None):
        print("Loading of set of molecules from {}...".format(file))
        super().__init__(file)
        with open(file, "r") as sdf:
            molecules_data = sdf.read()
        if molecules_data[-5:].strip() != "$$$$":
            exit(colored("{} is not valid sdf file. Last line is not $$$$.\n".format(sdf.name), "red"))
        molecules_data = [x.splitlines() for x in molecules_data.split("$$$$\n")]
        num_of_all_molecules = len(molecules_data) - 1
        if not num_of_molecules_to:
            self.num_of_molecules_to = num_of_all_molecules
        else:
            if num_of_molecules_to > num_of_all_molecules:
                exit(colored(("ERROR! Number of molecules is only {}!".format(num_of_all_molecules)), "red"))
            self.num_of_molecules_to = num_of_molecules_to
        if not num_of_molecules_from:
            self.num_of_molecules_from = 0
        else:
            self.num_of_molecules_from = num_of_molecules_from
        for molecule_data in molecules_data[self.num_of_molecules_from:self.num_of_molecules_to]:
            type_of_sdf_record = molecule_data[3][-5:]
            if type_of_sdf_record == "V2000":
                self.load_sdf_v2000(molecule_data)
            elif type_of_sdf_record == "V3000":
                self.load_sdf_v3000(molecule_data)
            else:
                exit(colored("{} is not valid sdf file.\n".format(sdf), "red"))
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

    def info(self, atomic_types_pattern):
        counter_atoms = Counter()
        counter_bonds = Counter()
        for molecule in self.molecules:
            counter_atoms.update(molecule.atoms_representation(atomic_types_pattern))
            counter_bonds.update(molecule.bonds_representation("{}_{}".format(atomic_types_pattern, atomic_types_pattern)))
        num_of_bonds = sum(counter_bonds.values())
        table_atoms = [(atom, count, round(count / (self.num_of_atoms / 100), 2)) for atom, count in counter_atoms.most_common()]
        table_bonds = [(bond, count, round(count / (num_of_bonds / 100), 2)) for bond, count in counter_bonds.most_common()]
        data = """Statistics data from set of molecules from {}
Number of molecules:   {}
Number of atoms:       {}
Number of atoms types: {}\n
{}\n\n
Number of bonds:       {}
Number of bonds types: {}\n
{}\n""".format(self.file, len(self.molecules), self.num_of_atoms, len(counter_atoms),
               tabulate(table_atoms, headers=["Type", "Number", "%"]), num_of_bonds, len(counter_bonds), tabulate(table_bonds, headers=["Type", "Number", "%"]))
        print(data)

    def add_ref_charges(self, file, num_of_atomic_types):
        with open(file, "r") as charges_file:
            names = [data.splitlines()[0] for data in charges_file.read().split("\n\n")[:-1]][self.num_of_molecules_from:self.num_of_molecules_to]
            control_order_of_molecules(names, [molecule.name for molecule in self.molecules], file, self.file)
        print("Loading charges from {}...".format(file))
        with open(file, "r") as charges_file:
            charges = []
            for molecule_data, molecule in zip(charges_file.read().split("\n\n")[:-1][self.num_of_molecules_from:self.num_of_molecules_to], self.molecules):
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
        for charge, symbolic_number in zip(self.ref_charges, self.all_symbolic_numbers_atoms):
            atomic_types_charges[symbolic_number].append(charge)
        self.ref_atomic_types_charges = array([array(chg, dtype=float32) for chg in atomic_types_charges])
        print(colored("ok\n", "green"))

    def create_method_data(self, method):
        self.all_num_of_atoms = array([molecule.num_of_atoms for molecule in self], dtype=int64)
        self.all_symbolic_numbers_atoms = concatenate([molecule.symbolic_numbers_atoms(method) for molecule in self], axis=0)
        if method.bond_types:
            self.all_symbolic_numbers_bonds = concatenate([molecule.symbolic_numbers_bonds(method) for molecule in self], axis=0)
        self.multiplied_all_symbolic_numbers_atoms = self.all_symbolic_numbers_atoms * len(method.atomic_parameters_types)
        for data in method.necessarily_data:
            setattr(self, "all_" + data, concatenate([getattr(molecule, data)() for molecule in self], axis=0))


class SubsetOfMolecules(SetOfMolecules):
    def __init__(self, original_set_of_molecules, method, subset):
        print("Creating of subset of molecules...")
        mol_chg = []
        index = 0
        for molecule in original_set_of_molecules:
            new_index = index + molecule.num_of_atoms
            mol_chg.append((molecule, original_set_of_molecules.ref_charges[index: new_index]))
            index = new_index
        counter_atoms = Counter()
        molecules = []
        ref_charges = []
        for molecule, charges in random.permutation(mol_chg):
            atoms = molecule.atoms_representation(method.atomic_types_pattern)
            if any(counter_atoms[atom] < subset for atom in atoms):
                counter_atoms.update(atoms)
                molecules.append(molecule)
                ref_charges.append(charges)
            if all(counter_atoms[x] > subset for x in method.atomic_types):
                break
        self.molecules = []
        self.ref_charges = []
        counter_atoms = Counter()
        for molecule, charges in zip(molecules[::-1], ref_charges[::-1]):
            atoms = molecule.atoms_representation(method.atomic_types_pattern)
            if any(counter_atoms[atom] < subset for atom in atoms):
                counter_atoms.update(atoms)
                self.molecules.append(molecule)
                self.ref_charges.append(charges)
            if all(counter_atoms[x] > subset for x in method.atomic_types):
                break
        if method.bond_types:
            counter_bonds = Counter()
            bond_format = "{}_{}".format(method.atomic_types_pattern, method.atomic_types_pattern)
            for molecule, charges in zip(self.molecules, self.ref_charges):
                bonds = molecule.bonds_representation(bond_format)
                counter_bonds.update(bonds)
            if len(counter_bonds) != len(method.bond_types):
                for molecule, charges in random.permutation(mol_chg):
                    bonds = molecule.bonds_representation(bond_format)
                    if any(counter_bonds[bond] < 1 for bond in bonds):
                        counter_bonds.update(bonds)
                        self.molecules.append(molecule)
                        self.ref_charges.append(charges)
                    if len(counter_bonds) == len(method.bond_types):
                        break
        self.ref_charges = array([y for x in self.ref_charges for y in x], dtype=float32)
        self.num_of_molecules = len(self.molecules)
        super().create_method_data(method)
        atomic_types_charges = [[] for _ in range(len(method.atomic_types))]
        for charge, symbolic_number in zip(self.ref_charges, self.all_symbolic_numbers_atoms):
            atomic_types_charges[symbolic_number].append(charge)
        self.ref_atomic_types_charges = array([array(chg, dtype=float32) for chg in atomic_types_charges])
        print("Subset of molecules contain {} molecules.".format(self.num_of_molecules))


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
                self.molecules.append(MoleculeChg(charges, molecule[0][0], atomic_types=atomic_symbols))
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
