from collections import Counter, defaultdict
from inspect import stack, getmodule
from sys import exit

import numpy as np
from numba.core.types import ListType, DictType, float32, int16, int64, string
from numba.experimental import jitclass
from numba.typed import List, Dict
from scipy.spatial.distance import cdist
from termcolor import colored

from .molecule import Molecule

array32 = float32[:]


@jitclass({"mols": ListType(Molecule.class_type.instance_type),
           "sdf_file": string,
           "ref_chgs_file": string,
           "emp_chgs_file": string,
           "num_of_mols": int64,
           "num_of_ats": int64,
           "ref_chgs": float32[:],
           "emp_chgs": float32[:],
           "emp_ats_types_chgs": DictType(string, float32[:]),
           "ref_ats_types_chgs": DictType(string, float32[:]),
           "all_ats_ids": int16[:],
           "ats_types": ListType(string),
           "bonds_types": ListType(string)})
class SetOfMolecules:
    def __init__(self, mols, sdf_file, ats_types, bonds_types):
        self.mols = mols
        self.sdf_file = sdf_file
        self.num_of_mols = len(self.mols)
        self.num_of_ats = np.sum(np.array([mol.num_of_ats for mol in self.mols], dtype=int64))
        self.ats_types = ats_types  # prepsat po fixu
        self.bonds_types = bonds_types  # prepsat po fixu

    def add_emp_chg(self,
                    chgs: np.array,
                    ats_types: list,
                    params_per_at_type: int):

        self.emp_chgs = chgs.astype(np.float32)
        self.emp_ats_types_chgs = Dict.empty(key_type=string, value_type=array32)
        for index, symbol in enumerate(ats_types):
            at_type_chg = chgs[self.all_ats_ids == index * params_per_at_type].astype(np.float32)
            if len(at_type_chg):
                self.emp_ats_types_chgs[symbol] = at_type_chg
        index = 0
        for mol in self.mols:
            mol.emp_chgs = chgs[index: index + mol.num_of_ats].astype(np.float32)
            index += mol.num_of_ats


def create_set_of_mols(sdf_file: str,
                       ats_types_pattern: str) -> SetOfMolecules:

    def load_sdf(mol_data: list) -> Molecule:

        def _create_at_bonds_representation() -> tuple:

            def _create_at_highest_bond() -> list:

                highest_bonds = [1] * num_of_ats
                for bonded_at_1, bonded_at_2, type in bonds:
                    if highest_bonds[bonded_at_1] < type:
                        highest_bonds[bonded_at_1] = type
                    if highest_bonds[bonded_at_2] < type:
                        highest_bonds[bonded_at_2] = type
                return List([symbol + "/" + str(hbo) for symbol, hbo in zip(symbols, highest_bonds)])

            def _create_at_bonded_at() -> list:
                bonded_ats = [[] for _ in range(num_of_ats)]
                for bonded_at1, bonded_at2, _ in bonds:
                    bonded_ats[bonded_at1].append(symbols[bonded_at2])
                    bonded_ats[bonded_at2].append(symbols[bonded_at1])
                return List([f"{symbol}/{''.join(sorted(bonded_ats))}"
                             for symbol, bonded_ats in zip(symbols, bonded_ats)])

            # create atoms_representation
            # plain (plain atom) = C, H, N, ...
            # hbo (highest bond order) = C/1, C/2, H/1, N/3, ...
            # plain-ba (plain atom + bonded atoms) = C/CCHH, H/N, ...
            # plain-ba-sb (plain atom + bonded atoms + simple bond definition) = C/CCHH, H/N, ...
            if ats_types_pattern == "plain":
                ats_srepr = List([at for at in symbols])
            elif ats_types_pattern == "hbo":
                ats_srepr = _create_at_highest_bond()
            elif ats_types_pattern in ["plain-ba", "plain-ba-sb"]:
                ats_srepr = _create_at_bonded_at()

            # create bonds_representation
            # plain (plain atom) = C-H-1, C-N-2, ...
            # hbo (highest bond order) = C/1-H/1-1, C/2-N/2-2, ...
            # plain-ba (plain atom + bonded atoms) = C/CCHH-H/C-1,...
            # plain-ba-sb (plain atom + bonded atoms + simple bond definition) = C-H-1, C-N-2, ... (same as plain)
            if ats_types_pattern == "plain-ba-sb":
                bonds_srepr = List([f"{'-'.join(sorted([symbols[bonded_at_1], symbols[bonded_at_2]]))}-{bond_type}"
                                    for bonded_at_1, bonded_at_2, bond_type in bonds])
            else:
                bonds_srepr = List([f"{'-'.join(sorted([ats_srepr[ba1], ats_srepr[ba2]]))}-{bond_type}"
                                    for ba1, ba2, bond_type in bonds])
            return ats_srepr, bonds_srepr

        def _sort(a: int,
                  b: int) -> tuple:
            if a > b:
                return b - 1, a - 1
            return a - 1, b - 1

        sdf_type = mol_data[3][-5:]
        if sdf_type == "V2000":
            name = mol_data[0]
            info_line = mol_data[3]
            num_of_ats = int(info_line[:3])
            num_of_bonds = int(info_line[3:6])
            symbols, ats_coordinates, bonds = [], [], []

            # read atoms lines
            for at_line in mol_data[4: num_of_ats + 4]:
                c1, c2, c3, symbol = at_line.split()[:4]
                ats_coordinates.append([float(c1), float(c2), float(c3)])
                symbols.append(symbol)

            # read bond lines
            for bond_line in mol_data[num_of_ats + 4: num_of_ats + num_of_bonds + 4]:
                a1, a2 = _sort(int(bond_line[:3]), int(bond_line[3:6]))
                bonds.append((a1, a2, int(bond_line[8])))

        elif sdf_type == "V3000":
            name = mol_data[0]
            info_line = mol_data[5].split()
            num_of_ats = int(info_line[3])
            num_of_bonds = int(info_line[4])
            symbols, ats_coordinates, bonds = [], [], []

            # read atoms lines
            for at_line in mol_data[7: num_of_ats + 7]:
                line = at_line.split()
                ats_coordinates.append(np.array([float(line[4]), float(line[5]), float(line[6])], dtype=np.float32))
                symbols.append(line[3])

            # read bond lines
            for bond_line in mol_data[num_of_ats + 9: num_of_ats + num_of_bonds + 9]:
                line = bond_line.split()
                at1, at2 = _sort(int(line[4]), int(line[5]))
                bonds.append((at1, at2, int(line[3])))

        ats_srepr, bonds_srepr = _create_at_bonds_representation()

        return Molecule(name,
                        num_of_ats,
                        np.array(ats_coordinates, dtype=np.float32),
                        ats_srepr,
                        np.array(bonds, dtype=np.int32),
                        bonds_srepr)

    print(f"Loading of set of molecules from {sdf_file}...")
    mols_data = [x.splitlines() for x in open(sdf_file, "r").read().split("$$$$\n")][:-1]
    mols = List([load_sdf(mol_data)
                 for mol_data in mols_data])
    ats_types = List(sorted(set([at for mol in mols for at in mol.ats_srepr]))) # prepsat po fixu
    bonds_types = List(sorted(set([bond for mol in mols for bond in mol.bonds_srepr]))) # prepsat po fixu
    set_of_mols = SetOfMolecules(mols,
                                 sdf_file,
                                 ats_types,  # prepsat po fixu
                                 bonds_types) # prepsat po fixu
    print(f"    {set_of_mols.num_of_mols} molecules was loaded.")
    print(colored("ok\n", "green"))
    return set_of_mols


def set_of_mols_info(sdf_file: str,
                     ats_types_pattern: str):

    set_of_mols = create_set_of_mols(sdf_file, ats_types_pattern)
    ats_in_mols = [mol.num_of_ats for mol in set_of_mols.mols]
    print(f"\nAtomic types pattern: {ats_types_pattern}")
    print(f"File:                 {sdf_file}\n")
    print(f"Number of mols:  {set_of_mols.num_of_mols}")
    print(f"Atoms in mols:   {min(ats_in_mols)} - {max(ats_in_mols)}")
    print(f"Number of atoms: {set_of_mols.num_of_ats}")
    print("\nAtomic types:")
    for at_type, number in Counter([at for mol in set_of_mols.mols for at in mol.ats_srepr]).most_common():
        print(f"    {at_type:>8}: {number:<11}")
    print("")


def create_method_data(method: "ChargeMethod",
                       set_of_mols: SetOfMolecules):

    for mol in set_of_mols.mols:
        mol.ats_ids = np.array([method.ats_types.index(at_type)
                                for at_type in mol.ats_srepr],
                               dtype=np.int16) * len(method.params["atom"]["names"])
        mol.distance_matrix = cdist(mol.ats_coordinates, mol.ats_coordinates).astype(np.float32)

    if "bond" in method.params:
        for mol in set_of_mols.mols:
            mol.bonds_ids = np.array([method.bond_types.index(bond)
                                      for bond in mol.bonds_srepr],
                                     dtype=np.int16) + len(method.params["atom"]["names"]) * len(method.ats_types)

    if "parameterization.py" in getmodule(stack()[1][0]).__file__:
        _create_chgs_attr(set_of_mols, "ref_chgs")
        set_of_mols.all_ats_ids = np.array([at_id for molecule in set_of_mols.mols
                                            for at_id in molecule.ats_ids], dtype=np.int16)



def create_80_20(set_of_mols: SetOfMolecules):  # smazat!!!
    percent_index_80 = int((set_of_mols.num_of_mols/100)*80)
    mols = np.random.permutation(set_of_mols.mols)
    mols_80 = mols[:percent_index_80]
    mols_20 = mols[percent_index_80:]

    ats_types_80 = List(sorted(set([at for mol in mols_80 for at in mol.ats_srepr])))  # po fixu přepsat
    bonds_types_80 = List(sorted(set([bond for mol in mols_80 for bond in mol.bonds_srepr])))  # po fixu přepsat
    set_of_mols_80 = SetOfMolecules(List(mols_80), set_of_mols.sdf_file, ats_types_80, bonds_types_80)

    set_of_mols_80.ref_chgs_file = set_of_mols.ref_chgs_file
    set_of_mols_80.emp_chgs_file = set_of_mols.emp_chgs_file

    ats_types_20 = List(sorted(set([at for mol in mols_20 for at in mol.ats_srepr])))  # po fixu přepsat
    bonds_types_20 = List(sorted(set([bond for mol in mols_20 for bond in mol.bonds_srepr])))  # po fixu přepsat
    set_of_mols_20 = SetOfMolecules(List(mols_20), set_of_mols.sdf_file, ats_types_20, bonds_types_20)

    set_of_mols_20.ref_chgs_file = set_of_mols.ref_chgs_file
    set_of_mols_20.emp_chgs_file = set_of_mols.emp_chgs_file

    return set_of_mols_80, set_of_mols_20

def create_par_val_set(set_of_mols: SetOfMolecules,
                       subset: int,
                       chg_method: "ChargeMethod") -> tuple:

    def _create_subset_of_mols(subset: int) -> SetOfMolecules:

        def _select_molecules(mols: List,
                              subset: int) -> list:

            counter_at = Counter()
            subset_mols = []
            for mol in mols:
                if any(counter_at[at] < subset for at in mol.ats_srepr):
                    counter_at.update(mol.ats_srepr)
                    subset_mols.append(mol)
                if all(counter_at[x] > subset for x in chg_method.ats_types):
                    break
            return subset_mols

        subset_mols = _select_molecules(np.random.permutation(set_of_mols.mols), subset)
        subset_mols = _select_molecules(subset_mols[::-1], subset)

        if "bond" in chg_method.params:
            counter_bonds = Counter()
            for mol in subset_mols:
                counter_bonds.update(mol.bonds_srepr)

            if any(counter_bonds[bond] < subset // 5 + 1 for bond in set_of_mols.bonds_types):
                mols_names = [mol.name for mol in subset_mols]
                for mol in set_of_mols.mols:
                    bonds = mol.bonds_srepr
                    if mol.name not in mols_names and any(counter_bonds[bond] < subset // 5 + 1 for bond in bonds):
                        counter_bonds.update(bonds)
                        subset_mols.append(mol)
                        mols_names.append(mol.name)
                    if all(counter_bonds[bond] >= subset // 5 + 1 for bond in set_of_mols.bonds_types):
                        break

        numba_mols = List(subset_mols)

        at_types_n = List(sorted(set([at for mol in numba_mols for at in mol.ats_srepr])))  # prepsat po fixu
        bonds_types_n = List(sorted(set([bond for mol in numba_mols for bond in mol.bonds_srepr])))  # prepsat po fixu

        subset_of_mol = SetOfMolecules(numba_mols, set_of_mols.sdf_file, at_types_n, bonds_types_n)
        return subset_of_mol

    print("Creating validation and parameterization sets...")
    set_of_mols_par = _create_subset_of_mols(subset)
    set_of_mols_min = _create_subset_of_mols(5)
    if len(set_of_mols_par.mols) == len(set_of_mols.mols):
        exit(colored("Error! It is too small set of molecules or too high --subset value.\n", "red"))

    set_of_mols_par.ref_chgs_file = set_of_mols.ref_chgs_file
    set_of_mols_par.emp_chgs_file = set_of_mols.emp_chgs_file

    par_mols_names = [molecule.name for molecule in set_of_mols_par.mols]
    mols_val = List([mol for mol in set_of_mols.mols if mol.name not in par_mols_names])
    ats_types_val = List(sorted(set([at for mol in mols_val for at in mol.ats_srepr]))) # po fixu přepsat
    bonds_types_val = List(sorted(set([bond for mol in mols_val for bond in mol.bonds_srepr])))  # po fixu přepsat
    set_of_mols_val = SetOfMolecules(mols_val, set_of_mols.sdf_file, ats_types_val, bonds_types_val)
    print(f"    {set_of_mols_par.num_of_mols} molecules in parameterization set.")
    print(f"    {set_of_mols_min.num_of_mols} molecules in minimum set.")
    print(f"    {set_of_mols_val.num_of_mols} molecules in validation set.")
    print(colored("ok\n", "green"))

    return set_of_mols_par, set_of_mols_min, set_of_mols_val


def _create_chgs_attr(set_of_mols: SetOfMolecules,
                      type_of_chg: str):

    setattr(set_of_mols, type_of_chg, np.array([at_chg for mol in set_of_mols.mols
                                                for at_chg in getattr(mol, type_of_chg)], dtype=np.float32))
    ats_types_chg = defaultdict(list)
    for chg, symbol in zip(getattr(set_of_mols, type_of_chg),
                           [symbol for mol in set_of_mols.mols for symbol in mol.ats_srepr]):
        ats_types_chg[symbol].append(chg)
    numba_at_types_chg = Dict.empty(key_type=string, value_type=float32[:])
    for at_type, value in ats_types_chg.items():
        numba_at_types_chg[at_type] = np.array(value, dtype=np.float32)
    setattr(set_of_mols, f"{type_of_chg[:3]}_ats_types_chgs", numba_at_types_chg)


def add_chgs(set_of_mols: SetOfMolecules,
             chgs_file: str,
             type_of_chgs: str):

    def _control_order_of_mol():
        print("    Control of order of molecules... ")
        sdf_names = [mol.name for mol in set_of_mols.mols]
        chgs_names = [data.splitlines()[0] for data in chgs_file.read().split("\n\n")[:-1]][:set_of_mols.num_of_mols]
        if sdf_names == chgs_names:
            return True
        sdf_names = set(sdf_names)
        chgs_names = set(chgs_names)
        if sdf_names == chgs_names:
            exit(colored(f"Files {set_of_mols.sdf_file} and {chgs_file.name} contain same molecules, "
                         f"but in different order!\n", "red"))
        else:
            intersection = sdf_names.intersection(chgs_names)
            difference = sdf_names.symmetric_difference(chgs_names)
            print(colored(f"Files {set_of_mols.sdf_file} and {chgs_file.name} contain different set of molecules!"
                          f"    Number of common molecules:    {len(intersection)}"
                          f"    Number of different molecules: {len(difference)}", "red"))
            if input("Do you want to print difference molecules names? yes/no: ") == "yes":
                for name in difference:
                    print(name)
            exit("")

    setattr(set_of_mols, f"{type_of_chgs}_file", chgs_file)
    with open(chgs_file, "r") as chgs_file:
        print(f"Loading {'empirical' if type_of_chgs == 'emp_chgs' else 'reference'} charges from {chgs_file.name}...")
        _control_order_of_mol()
        chgs_file.seek(0)
        for mol_data, mol in zip(chgs_file.read().split("\n\n")[:-1], set_of_mols.mols):
            setattr(mol, type_of_chgs, np.array([float(line.split()[2])
                                                 for line in mol_data.splitlines()[2:]], dtype=np.float32))
    _create_chgs_attr(set_of_mols, type_of_chgs)
    print(colored("ok\n", "green"))


def write_chgs_to_file(chgs: np.array,
                       set_of_mol: SetOfMolecules):

    print(f"Writing charges to {set_of_mol.emp_chgs_file}...")
    with open(set_of_mol.emp_chgs_file, "w") as chgs_file:
        count = 0
        for mol in set_of_mol.mols:
            chgs_file.write(f"{mol.name}\n{mol.num_of_ats}\n")
            for index, at in enumerate(mol.ats_srepr):
                chgs_file.write("{0:>3} {1:>3} {2:>15}\n".format(index + 1,
                                                                 at.split("/")[0],
                                                                 str(float("{0:.6f}".format(chgs[count])))))
                count += 1
            chgs_file.write("\n")
    print(colored("ok\n", "green"))
