from .molecule import Molecule, MoleculeChg
from .control_order_of_molecules import control_order_of_molecules
from sys import exit
from termcolor import colored
from collections import Counter
from tabulate import tabulate
from numpy import array, float32, int32, concatenate, random, mean
from copy import copy
from inspect import stack


def sort(a, b):
    if a > b:
        return b-1, a-1
    return a-1, b-1


class ArciSet:

    def __len__(self):
        return self.num_of_molecules

    def __getitem__(self, index):
        return self.molecules[index]

    def __iter__(self):
        return iter(self.molecules)

    def create_indices(self, nums_of_atoms):
        if self.submolecules:
            indices = []
            index = 0
            for num_of_atom in nums_of_atoms:
                indices.append(index)
                index += num_of_atom
            self.indices = array(indices)
        else:
            self.indices = None


class SetOfMolecules(ArciSet):
    def __init__(self, sdf_file, num_of_molecules=None, parameterization=None, submolecules=None):
        self.submolecules = submolecules
        self.molecules = []
        self.file = sdf_file
        print(f"Loading of set of molecules from {self.file}...")
        molecules_data = open(self.file, "r").read()
        self.num_of_molecules = self.submolecules if self.submolecules else num_of_molecules if num_of_molecules else molecules_data.count("$$$$")
        if molecules_data[-5:].strip() != "$$$$":
            exit(colored(f"{self.file} is not valid sdf file. Last line does not contain $$$$.\n", "red"))
        molecules_data = [x.splitlines() for x in molecules_data.split("$$$$\n")][:self.submolecules if self.submolecules else self.num_of_molecules]
        for molecule_data in molecules_data:
            type_of_sdf_record = molecule_data[3][-5:]
            if type_of_sdf_record == "V2000":
                self.load_sdf_v2000(molecule_data)
            elif type_of_sdf_record == "V3000":
                self.load_sdf_v3000(molecule_data)
            else:
                exit(colored(f"{self.file} is not valid sdf file.\n", "red"))
        self.num_of_atoms = sum([len(molecule) for molecule in self.molecules])
        print(f"    {self.num_of_molecules} molecules was loaded.")
        print(colored("ok\n", "green"))

        if parameterization:
            ref_chg_file, validation = parameterization
            with open(ref_chg_file, "r") as reference_charges_file:
                names = [data.splitlines()[0] for data in reference_charges_file.read().split("\n\n")[:-1]][:num_of_molecules]
                sdf_names = []
                for molecule in self.molecules:
                    molecule_name = molecule.name.split("~~~")[0]
                    if molecule_name not in sdf_names:
                        sdf_names.append(molecule_name)
                self.original_sdf_names = sdf_names
                control_order_of_molecules(names, sdf_names, ref_chg_file, self.file)
                self.original_order_molecules_names = names
                print(f"Loading charges from {ref_chg_file}...")
                reference_charges_file.seek(0)
                if self.submolecules:
                    for molecule, molecule_charge in zip(self.molecules, [atom_charge.split()[2] for molecule_charges in reference_charges_file.read().split("\n\n")[:-1] for atom_charge in molecule_charges.splitlines()[2:]]):
                        molecule.charges = array([float(molecule_charge)])
                else:
                    for molecule_data, molecule in zip(reference_charges_file.read().split("\n\n")[:-1], self.molecules):
                        molecule_charges = []
                        for line in molecule_data.splitlines()[2:]:
                            molecule_charges.append(float(line.split()[2]))
                        molecule.charges = array(molecule_charges)
            print(colored("ok\n", "green"))
            print("Creating validation and parameterization sets...")
            original_molecules = self.molecules
            self.molecules = random.permutation(self.molecules)
            num_of_molecules_par_val = int((1 - validation / 100) * self.num_of_molecules)
            self.validation = copy(self)
            self.validation.molecules = self.validation.molecules[num_of_molecules_par_val:]
            self.validation.num_of_molecules = len(self.validation.molecules)
            self.parameterization = copy(self)
            self.parameterization.molecules = self.parameterization.molecules[:num_of_molecules_par_val]
            self.parameterization.num_of_molecules = len(self.parameterization.molecules)
            self.molecules = original_molecules
            print(f"    {len(self.parameterization.molecules)} molecules in parameterization set.")
            print(f"    {len(self.validation.molecules)} molecules in validation set.")
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

    def create_method_data(self, method):
        self.num_of_molecules = len(self.molecules)
        self.all_num_of_atoms = array([molecule.num_of_atoms for molecule in self], dtype=int32)
        self.all_symbolic_numbers_atoms = concatenate([molecule.symbolic_numbers_atoms(method) for molecule in self], axis=0)
        if method.bond_types:
            self.all_symbolic_numbers_bonds = concatenate([molecule.symbolic_numbers_bonds(method) for molecule in self], axis=0)
        self.multiplied_all_symbolic_numbers_atoms = self.all_symbolic_numbers_atoms * len(method.atomic_parameters_types)
        for data in method.necessarily_data:
            setattr(self, "all_" + data, concatenate([getattr(molecule, data)() for molecule in self], axis=0))
        if stack()[1][0].f_locals["self"].__class__.__name__ in ["Parameterization", "SubsetOfMolecules"]:
            self.ref_charges = array([at_chg for molecule in self.molecules for at_chg in molecule.charges])
            atomic_types_charges = [[] for _ in range(len(method.atomic_types))]
            if self.submolecules:
                self.create_indices(self.all_num_of_atoms)
                self.all_symbolic_numbers_atoms_submolecules = self.all_symbolic_numbers_atoms[self.indices]
            else:
                self.indices = None
            for charge, symbolic_number in zip(self.ref_charges, self.all_symbolic_numbers_atoms if not self.submolecules else self.all_symbolic_numbers_atoms_submolecules):
                atomic_types_charges[symbolic_number].append(charge)
            self.ref_atomic_types_charges = array([array(chg, dtype=float32) for chg in atomic_types_charges])




def select_molecules(all_molecules, subset, method, submolecules):
    counter_atoms = Counter()
    molecules = []
    if submolecules:
        for molecule in all_molecules:
            atom = molecule.atoms_representation(method.atomic_types_pattern)[0]
            if counter_atoms[atom] < subset:
                counter_atoms.update((atom,))
                molecules.append(molecule)
            if all(counter_atoms[x] > subset for x in method.atomic_types):
                break
    else:
        for molecule in all_molecules:
            atoms = molecule.atoms_representation(method.atomic_types_pattern)
            if any(counter_atoms[atom] < subset for atom in atoms):
                counter_atoms.update(atoms)
                molecules.append(molecule)
            if all(counter_atoms[x] > subset for x in method.atomic_types):
                break
    return molecules


class SubsetOfMolecules(SetOfMolecules):
    def __init__(self, original_set_of_molecules, method, subset, submolecules):
        self.submolecules = submolecules
        molecules = select_molecules(random.permutation(original_set_of_molecules.molecules), subset, method, self.submolecules)
        self.molecules = select_molecules(molecules[::-1], subset, method, self.submolecules)
        if method.bond_types:
            counter_bonds = Counter()
            bond_format = "{}_{}".format(method.atomic_types_pattern, method.atomic_types_pattern)
            for molecule in original_set_of_molecules.molecules:
                bonds = molecule.bonds_representation(bond_format)
                counter_bonds.update(bonds)
            if len(counter_bonds) != len(method.bond_types):
                for molecule in original_set_of_molecules.molecules:
                    bonds = molecule.bonds_representation(bond_format)
                    if any(counter_bonds[bond] < 1 for bond in bonds):
                        counter_bonds.update(bonds)
                        self.molecules.append(molecule)
                    if len(counter_bonds) == len(method.bond_types):
                        break
        super().create_method_data(method)


class SetOfMoleculesFromChargesFile(ArciSet):
    def __init__(self, file, ref=True):
        print("Loading of set of molecules from {}...".format(file))
        self.file = file
        self.molecules = []
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
