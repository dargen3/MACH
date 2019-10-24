from .molecule import Molecule, MoleculeChg
from .control_order_of_molecules import control_order_of_molecules
from sys import exit
from termcolor import colored
from collections import Counter
from tabulate import tabulate
from numpy import array, float32, int32, concatenate, random, mean
from copy import copy
from inspect import stack
from .atom import Atom
from .bond import Bond




def create_atom_high_bond(num_of_atoms, bonds, atomic_symbols):
    highest_bonds = [1] * num_of_atoms
    for bond in bonds:
        ba1i = bond.atom1.index
        ba2i = bond.atom2.index
        type = bond.type_of_bond
        if highest_bonds[ba1i] < type:
            highest_bonds[ba1i] = type
        if highest_bonds[ba2i] < type:
            highest_bonds[ba2i] = type
    return ["{}~{}".format(a, b) for a, b in zip(atomic_symbols, highest_bonds)]

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



class SetOfMolecules(ArciSet):
    def __init__(self, sdf_file, num_of_molecules=None, parameterization=None, random_seed=0, molecules_list=False):
        if molecules_list:
            self.molecules = sdf_file
            self.num_of_molecules = len(self.molecules)
        else:
            random.seed(random_seed)
            self.molecules = []
            self.file = sdf_file
            print(f"Loading of set of molecules from {self.file}...")
            molecules_data = open(self.file, "r").read()

            molecules_in_file = molecules_data.count("$$$$")


            self.num_of_molecules = num_of_molecules if num_of_molecules else molecules_in_file
            if self.num_of_molecules > molecules_data.count("$$$$"):
                exit(colored(f"Error! There is only {molecules_in_file} molecules in {self.file}.", "red"))


            if molecules_data[-5:].strip() != "$$$$":
                exit(colored(f"Error! {self.file} is not valid sdf file. Last line does not contain $$$$.\n", "red"))
            molecules_data = [x.splitlines() for x in molecules_data.split("$$$$\n")][:self.num_of_molecules]
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
            sdf_names = []
            for molecule in self.molecules:
                molecule_name = molecule.name.split("~~~")[0]
                if molecule_name not in sdf_names:
                    sdf_names.append(molecule_name)
            self.original_sdf_names = sdf_names
            print(colored("ok\n", "green"))
            if parameterization:
                ref_chg_file, self.validation_percent = parameterization
                with open(ref_chg_file, "r") as reference_charges_file:
                    names = [data.splitlines()[0] for data in reference_charges_file.read().split("\n\n")[:-1]][:num_of_molecules]
                    control_order_of_molecules(names, self.original_sdf_names, ref_chg_file, self.file)
                    self.original_order_molecules_names = names
                    print(f"Loading charges from {ref_chg_file}...")
                    reference_charges_file.seek(0)
                    for molecule_data, molecule in zip(reference_charges_file.read().split("\n\n")[:-1], self.molecules):
                        molecule_charges = []
                        for line in molecule_data.splitlines()[2:]:
                            molecule_charges.append(float(line.split()[2]))
                        molecule.charges = array(molecule_charges)
                print(colored("ok\n", "green"))
                print("Creating validation and parameterization sets...")
                original_molecules = self.molecules
                self.molecules = random.permutation(self.molecules)
                num_of_molecules_par_val = int((1 - self.validation_percent / 100) * (self.num_of_molecules))
                if num_of_molecules_par_val == 0:
                    num_of_molecules_par_val = 1
                self.validation = copy(self)
                self.parameterization = copy(self)
                self.validation.molecules = self.validation.molecules[num_of_molecules_par_val:]
                self.parameterization.molecules = self.parameterization.molecules[:num_of_molecules_par_val]
                self.validation.num_of_molecules = len(self.validation.molecules)
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
            atomic_coordinates.append(array([float(line[0]), float(line[1]), float(line[2])], dtype=float32))
            atomic_symbols.append(line[3])



        atoms = [Atom(cor, atomic_symbol,index) for index, (cor, atomic_symbol) in enumerate(zip(atomic_coordinates, atomic_symbols))]




        for bond_line in molecular_data[num_of_atoms + 4: num_of_atoms + num_of_bonds + 4]:
            # bonds.append((sort(int(bond_line[:3]), int(bond_line[3:6])), int(bond_line[8])))
            a1, a2 = sort(int(bond_line[:3]), int(bond_line[3:6]))
            bonds.append(Bond(atoms[a1], atoms[a2], int(bond_line[8])))


        for atom, atom_high_bond in zip(atoms, create_atom_high_bond(num_of_atoms, bonds, atomic_symbols)):
            atom.hbo = atom_high_bond



        self.molecules.append(Molecule(name, atoms, bonds))

    def load_sdf_v3000(self, molecular_data):
        name = molecular_data[0]
        info_line = molecular_data[5].split()
        num_of_atoms = int(info_line[3])
        num_of_bonds = int(info_line[4])
        atomic_symbols, atomic_coordinates, bonds = [], [], []
        for atom_line in molecular_data[7: num_of_atoms + 7]:
            line = atom_line.split()
            atomic_coordinates.append(array([float(line[4]), float(line[5]), float(line[6])], dtype=float32))
            atomic_symbols.append(line[3])

        atoms = [Atom(cor, atomic_symbol, index) for index, (cor, atomic_symbol) in enumerate(zip(atomic_coordinates, atomic_symbols))]



        for bond_line in molecular_data[num_of_atoms + 9: num_of_atoms + num_of_bonds + 9]:
            line = bond_line.split()
            a1, a2 = sort(int(line[4]), int(line[5]))
            bonds.append(Bond(atoms[a1], atoms[a2], int(line[3])))


        for atom, atom_high_bond in zip(atoms, create_atom_high_bond(num_of_atoms, bonds, atomic_symbols)):
            atom.hbo = atom_high_bond


        self.molecules.append(Molecule(name, atoms, bonds))

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




        try:
            self.all_symbolic_numbers_atoms = concatenate([molecule.symbolic_numbers_atoms(method) for molecule in self], axis=0)
            if method.bond_types:
                self.all_symbolic_numbers_bonds = concatenate([molecule.symbolic_numbers_bonds(method) for molecule in self], axis=0)
        except ValueError as e:
            exit(colored("Error! {} is not in last {}% of set of molecules used for validation!\n".format(str(e).split("'")[1], self.validation_percent), "red"))





        self.multiplied_all_symbolic_numbers_atoms = self.all_symbolic_numbers_atoms * len(method.atomic_parameters_types)
        for data in method.necessarily_data:
            setattr(self, "all_" + data, concatenate([getattr(molecule, data)() for molecule in self], axis=0))

        if stack()[1][0].f_locals["self"].__class__.__name__ in ["Parameterization", "SubsetOfMolecules"]:
            self.ref_charges = array([at_chg for molecule in self.molecules for at_chg in molecule.charges])
            atomic_types_charges = [[] for _ in range(len(method.atomic_types))]
            for charge, symbolic_number in zip(self.ref_charges, self.all_symbolic_numbers_atoms):
                atomic_types_charges[symbolic_number].append(charge)
            self.ref_atomic_types_charges = array([array(chg, dtype=float32) for chg in atomic_types_charges])



def select_molecules(all_molecules, subset, method):
    counter_atoms = Counter()
    molecules = []
    for molecule in all_molecules:
        atoms = molecule.atoms_representation(method.atomic_types_pattern)
        if any(counter_atoms[atom] < subset for atom in atoms):
            counter_atoms.update(atoms)
            molecules.append(molecule)
        if all(counter_atoms[x] > subset for x in method.atomic_types):
            break
    return molecules


class SubsetOfMolecules(SetOfMolecules):
    def __init__(self, original_set_of_molecules, method, subset):
        molecules = select_molecules(random.permutation(original_set_of_molecules.molecules), subset, method)
        self.molecules = select_molecules(molecules[::-1], subset, method)
        if method.bond_types:
            counter_bonds = Counter()
            bond_format = "{}_{}".format(method.atomic_types_pattern, method.atomic_types_pattern)
            for molecule in self.molecules:
                counter_bonds.update(molecule.bonds_representation(bond_format))
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
