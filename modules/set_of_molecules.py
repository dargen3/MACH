from collections import Counter, defaultdict
from inspect import stack, getmodule
from sys import exit

import numpy as np
from numba.core.types import ListType, DictType, float32, float64, int16, int64, string
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
           "num_of_bonds": int64,
           "ref_chgs": float32[:],
           "emp_chgs": float32[:],
           "ref_av_charges": float64[:],  # mozna smazat!
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
        self.num_of_bonds = np.sum(np.array([mol.num_of_bonds for mol in self.mols], dtype=int64))
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

            def _create_hbo_ba() -> list:
                bonded_ats = [[] for _ in range(num_of_ats)]
                for bonded_at1, bonded_at2, _ in bonds:
                    bonded_ats[bonded_at1].append(symbols[bonded_at2])
                    bonded_ats[bonded_at2].append(symbols[bonded_at1])
                return List([f"{symbol}/{''.join(sorted(bonded_ats))}"
                             for symbol, bonded_ats in zip(_create_at_highest_bond(), bonded_ats)])

            def _create_at_bonded_at_bonded_at() -> list:
                bonded_ats = [[] for _ in range(num_of_ats)]
                for bonded_at1, bonded_at2, _ in bonds:
                    bonded_ats[bonded_at1].append(symbols[bonded_at2])
                    bonded_ats[bonded_at2].append(symbols[bonded_at1])

                bonded_bonded_ats = [[] for _ in range(num_of_ats)]
                for bonded_at1, bonded_at2, _ in bonds:
                    bonded_bonded_ats[bonded_at1].extend(bonded_ats[bonded_at2])
                    bonded_bonded_ats[bonded_at1].remove(symbols[bonded_at1])
                    bonded_bonded_ats[bonded_at2].extend(bonded_ats[bonded_at1])
                    bonded_bonded_ats[bonded_at2].remove(symbols[bonded_at2])

                return List([f"{symbol}/{''.join(sorted(bonded_ats))}/{''.join(sorted(bonded_bonded_ats))}"
                             for symbol, bonded_ats, bonded_bonded_ats in zip(symbols, bonded_ats, bonded_bonded_ats)])

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
            elif ats_types_pattern == "hbo-ba":
                ats_srepr = _create_hbo_ba()
            elif ats_types_pattern == "plain-ba-hbo":
                ats_srepr = List([plain_ba_srepr + "..." + hbo_srepr for plain_ba_srepr, hbo_srepr in zip(_create_at_bonded_at(), _create_at_highest_bond())])
            elif ats_types_pattern == "plain-ba-ba":
                ats_srepr = _create_at_bonded_at_bonded_at()
            #
            # if ats_types_pattern == "plain-ba":
            #     for bonded_at1, bonded_at2, _ in bonds:
            #         if ats_srepr[bonded_at1] == "O/C" and ats_srepr[bonded_at2] == "C/COO":
            #             ats_srepr[bonded_at1] = "O/Cx"
            #         elif ats_srepr[bonded_at2] == "O/C" and ats_srepr[bonded_at1] == "C/COO":
            #             ats_srepr[bonded_at2] = "O/Cx"


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

        def load_formal_charge_from_line(line: list) -> int: # asi upravit pro kovy a pro V3000
            if len(line) not in [9, 16]:
                return 0 # V3000
            chg = int(line[5])
            if chg != 0:
                chg = -int(chg)+4

            if chg == 0 and len(line) == 16:
                valence = line[9]
                if valence != "0":
                    try:
                        chg = {"N4":1, "O1": -1}[line[3]+valence]
                    except KeyError:
                        print(line)
                        return 0
            #
            # if chg != 0:
            #     print(chg, line)
            return chg




        sdf_type = mol_data[3][-5:]
        symbols, ats_coordinates, bonds, formal_chgs = [], [], [], []

        if sdf_type == "V2000":
            name = mol_data[0]
            info_line = mol_data[3]
            num_of_ats = int(info_line[:3])
            num_of_bonds = int(info_line[3:6])

            # read atoms lines
            for at_line in mol_data[4: num_of_ats + 4]:
                s_line = at_line.split()
                c1, c2, c3, symbol = s_line[:4]
                ats_coordinates.append([float(c1), float(c2), float(c3)])
                symbols.append(symbol)
                formal_chgs.append(load_formal_charge_from_line(s_line))

            # read bond lines
            for bond_line in mol_data[num_of_ats + 4: num_of_ats + num_of_bonds + 4]:
                a1, a2 = _sort(int(bond_line[:3]), int(bond_line[3:6]))
                bonds.append((a1, a2, int(bond_line[8])))

        elif sdf_type == "V3000":
            name = mol_data[0]
            info_line = mol_data[5].split()
            num_of_ats = int(info_line[3])
            num_of_bonds = int(info_line[4])

            # read atoms lines
            for at_line in mol_data[7: num_of_ats + 7]:
                line = at_line.split()
                ats_coordinates.append(np.array([float(line[4]), float(line[5]), float(line[6])], dtype=np.float32))
                symbols.append(line[3])
                formal_chgs.append(load_formal_charge_from_line(line))

            # read bond lines
            for bond_line in mol_data[num_of_ats + 9: num_of_ats + num_of_bonds + 9]:
                line = bond_line.split()
                at1, at2 = _sort(int(line[4]), int(line[5]))
                bonds.append((at1, at2, int(line[3])))

        ats_srepr, bonds_srepr = _create_at_bonds_representation()

        if not bonds_srepr: # rozhodně přepsat!!!! pro molekuly o jednom atomu (když molekula nemá žádnou vazbu)
            bonds_srepr = List(["none"])
            bonds= [[0,0,1]]

        return Molecule(name,
                        num_of_ats,
                        num_of_bonds,
                        np.array(ats_coordinates, dtype=np.float32),
                        ats_srepr,
                        np.array(bonds, dtype=np.int32),
                        bonds_srepr,
                        np.array(formal_chgs, dtype=np.float64))

    print(f"Loading of set of molecules from {sdf_file}...")
    mols_data = [x.splitlines() for x in open(sdf_file, "r").read().split("$$$$\n")][:-1]
    mols = List([load_sdf(mol_data)
                 for mol_data in mols_data])

    if ats_types_pattern == "plain-ba-hbo":
        c = Counter(at_srepr.split("...")[0] for mol in mols for at_srepr in mol.ats_srepr)


        cb_list = []
        for mol in mols:
            for bond_type in mol.bonds_srepr:
                a1, a2, type = bond_type.split("-")
                cb_list.append("-".join([a1.split("...")[0], a2.split("...")[0], type]))
        cb = Counter(cb_list)
        print(cb)

        for mol in mols:
            final = []
            for at_type in mol.ats_srepr:
                plain_ba_r, hbo_r = at_type.split("...")
                if c[plain_ba_r] > 10:
                    final.append(plain_ba_r)
                else:
                    final.append(hbo_r)
            mol.ats_srepr = List(final)

            final_b = []
            for bond_type in mol.bonds_srepr:
                a1, a2, type = bond_type.split("-")
                if cb["-".join([a1.split("...")[0], a2.split("...")[0], type])] > 10:
                    final_b.append("-".join([a1.split("...")[0], a2.split("...")[0], type]))
                else:
                    final_b.append("-".join([a1.split("...")[1], a2.split("...")[1], type]))
            mol.bonds_srepr = List(final_b)



    ats_types = List(sorted(set([at for mol in mols for at in mol.ats_srepr]))) # prepsat po fixu
    bonds_types = List(sorted(set([bond for mol in mols for bond in mol.bonds_srepr]))) # prepsat po fixu
    set_of_mols = SetOfMolecules(mols,
                                 sdf_file,
                                 ats_types,  # prepsat po fixu
                                 bonds_types)  # prepsat po fixu
    print(f"    {set_of_mols.num_of_mols} molecules was loaded.")
    print(colored("ok\n", "green"))
    return set_of_mols


def set_of_mols_info(sdf_file: str,
                     ats_types_pattern: str):

    set_of_mols = create_set_of_mols(sdf_file, ats_types_pattern)
    ats_in_mols = [mol.num_of_ats for mol in set_of_mols.mols]
    print(f"\nAtomic types pattern: {ats_types_pattern}")
    print(f"File:                   {sdf_file}\n")
    print(f"Number of molecules:    {set_of_mols.num_of_mols}")
    print(f"Atoms in molecules:     {min(ats_in_mols)} - {max(ats_in_mols)}")
    print(f"Number of atoms:        {set_of_mols.num_of_ats}")
    print(f"Number of bonds:        {set_of_mols.num_of_bonds}")
    print(f"Number of atomic types: {len(set_of_mols.ats_types)}")
    print(f"Number of bonds types:  {len(set_of_mols.bonds_types)}")

    print("\nAtomic types:")
    atoms_counter = Counter([at for mol in set_of_mols.mols for at in mol.ats_srepr]).most_common()
    for at_type, number in atoms_counter:
        if number < 5:
            mols_with_at_type = [mol.name for mol in set_of_mols.mols if at_type in mol.ats_srepr]
            print(f"    {at_type:>8}: {number:<11}    {'; '.join(mols_with_at_type)}")
        else:
            print(f"    {at_type:>8}: {number:<11}")

    print("")
    bonds_counter = Counter([bond for mol in set_of_mols.mols for bond in mol.bonds_srepr]).most_common()
    print("\n\nBonds types:")
    for bond_type, number in bonds_counter:
        if number < 20:
            mols_with_bond_type = [mol.name for mol in set_of_mols.mols if bond_type in mol.bonds_srepr]
            print(f"    {bond_type:>8}: {number:<11}    {'; '.join(mols_with_bond_type)}")
        else:
            print(f"    {bond_type:>8}: {number:<11}")
    print("")



def create_method_data(chg_method: "ChargeMethod",
                       set_of_mols: SetOfMolecules):

    for mol in set_of_mols.mols:
        mol.ats_ids = np.array([chg_method.ats_types.index(at_type)
                                for at_type in mol.ats_srepr],
                               dtype=np.int16) * len(chg_method.params["atom"]["names"])
        mol.distance_matrix = cdist(mol.ats_coordinates, mol.ats_coordinates).astype(np.float32)

    if "bond" in chg_method.params:
        for mol in set_of_mols.mols:
            mol.bonds_ids = np.array([chg_method.bond_types.index(bond)
                                      for bond in mol.bonds_srepr],
                                     dtype=np.int16) + len(chg_method.params["atom"]["names"]) * len(chg_method.ats_types)

    if "parameterization.py" in getmodule(stack()[1][0]).__file__:
        _create_chgs_attr(set_of_mols, "ref_chgs")
        set_of_mols.all_ats_ids = np.array([at_id for molecule in set_of_mols.mols
                                            for at_id in molecule.ats_ids], dtype=np.int16)



def create_80_20(set_of_mols: SetOfMolecules,
                 parameterize_percent: int,
                 seed: int):
    percent_index_80 = int((set_of_mols.num_of_mols/100)*parameterize_percent)


    np.random.seed(1)  #Odkomentovat!!!!!!
    mols = np.random.permutation(set_of_mols.mols)
    np.random.seed(seed if seed != 0 else None)
    mols_80 = mols[:percent_index_80]
    mols_20 = mols[percent_index_80:]

    # original_names = [mol.name for mol in set_of_mols.mols]
    #
    #
    # from os.path import basename
    # names_parameterization_m = []
    # with open(f"modules/data_clanek_smazat/{basename(set_of_mols.sdf_file).split('.')[0]}_parameterization.log", "r") as names_file:
    #     for line in names_file.readlines()[1:]:
    #         names_parameterization_m.append(line.split(",")[0])
    #
    # names_validation_m = []
    # with open(f"modules/data_clanek_smazat/{basename(set_of_mols.sdf_file).split('.')[0]}_validation.log", "r") as names_file:
    #     for line in names_file.readlines()[1:]:
    #         names_validation_m.append(line.split(",")[0])
    #
    # mols_80 = List([set_of_mols.mols[original_names.index(name)] for name in names_parameterization_m])
    # mols_20 = List([set_of_mols.mols[original_names.index(name)] for name in names_validation_m])
    # # potud smazat
    #







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
                       subset: int) -> tuple:

    def _create_subset_of_mols(subset: int) -> SetOfMolecules:

        def _select_molecules(mols: List,
                              subset: int) -> list:

            counter_at = Counter()
            counter_bonds = Counter()

            subset_mols = []
            for mol in mols:
                if any(counter_at[at] < subset for at in mol.ats_srepr) or any(counter_bonds[bond] < subset for bond in mol.bonds_srepr):
                    counter_at.update(mol.ats_srepr)
                    counter_bonds.update(mol.bonds_srepr)
                    subset_mols.append(mol)
                if all(counter_at[x] >= subset for x in set_of_mols.ats_types) and all(counter_bonds[bond] >= subset for bond in set_of_mols.bonds_types):
                    break
            return subset_mols

        mols_num_of_ats_bonds_types = []
        for mol in set_of_mols.mols:
            num_of_ats_bonds_types = len(set(mol.ats_srepr)) + len(set(mol.bonds_srepr))
            mols_num_of_ats_bonds_types.append((mol, num_of_ats_bonds_types/mol.num_of_ats))
        set_of_mols_mols = [mol[0] for mol in sorted(mols_num_of_ats_bonds_types, key=lambda x: x[1], reverse=True)]

        subset_mols = _select_molecules(set_of_mols_mols, subset)
        subset_mols = _select_molecules(subset_mols[::-1], subset)

        numba_mols = List(subset_mols)

        at_types_n = List(sorted(set([at for mol in numba_mols for at in mol.ats_srepr])))  # prepsat po fixu
        bonds_types_n = List(sorted(set([bond for mol in numba_mols for bond in mol.bonds_srepr])))  # prepsat po fixu

        subset_of_mol = SetOfMolecules(numba_mols, set_of_mols.sdf_file, at_types_n, bonds_types_n)
        return subset_of_mol

    print("Creating validation and parameterization sets...")
    subset_of_mols = _create_subset_of_mols(subset)
    min_subset_of_mols = _create_subset_of_mols(1)
    if len(subset_of_mols.mols) == len(set_of_mols.mols):
        exit(colored("Error! It is too small set of molecules or too high --subset value.\n", "red"))

    print(f"    {subset_of_mols.num_of_mols} molecules in subset.")
    print(f"    {min_subset_of_mols.num_of_mols} molecules in minimum subset.")
    print(colored("ok\n", "green"))

    return subset_of_mols, min_subset_of_mols


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
            print(colored(f"Files {set_of_mols.sdf_file} and {chgs_file.name} contain same molecules, "
                          f"but in different order!\n", "red"))
            if input(f"Do you want to sort molecules in {chgs_file.name}? yes/no: ") == "yes":
                chgs_mols = {}
                chgs_file.seek(0)
                for mol_data in chgs_file.read().split("\n\n")[:-1]:
                    chgs_mols[mol_data.splitlines()[0]] = mol_data
                chgs_file.seek(0)
                chgs_file.truncate()
                for mol in set_of_mols.mols:
                    chgs_file.write(chgs_mols[mol.name])
                    chgs_file.write("\n\n")
                chgs_file.write("\n")


            else:
                exit()



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
    with open(chgs_file, "r+") as chgs_file:
        print(f"Loading {'empirical' if type_of_chgs == 'emp_chgs' else 'reference'} charges from {chgs_file.name}...")
        _control_order_of_mol()
        chgs_file.seek(0)
        for mol_data, mol in zip(chgs_file.read().split("\n\n")[:-1], set_of_mols.mols):
            setattr(mol, type_of_chgs, np.array([float(line.split()[2])
                                                 for line in mol_data.splitlines()[2:]], dtype=np.float32))
    _create_chgs_attr(set_of_mols, type_of_chgs)
    print(colored("ok\n", "green"))

    # print("test")
    # for molecule in set_of_mols.mols:
    #     diff = abs(molecule.total_chg-sum(molecule.formal_chgs))
    #
    #     print(molecule.total_chg, molecule.formal_chgs, diff)
    #     if diff > 0.01:
    #         print("ERRRORRRR")
    #         exit()

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
