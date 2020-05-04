from collections import Counter, defaultdict
from inspect import stack, getmodule
from sys import exit

from numba.experimental import jitclass
from numba.typed import List, Dict
from numba.core.types import ListType, DictType, float32, int16, int64, string
from numpy import array, concatenate, random, sum
from numpy import float32 as npfloat32
from numpy import int16 as npint16
from scipy.spatial.distance import cdist
from termcolor import colored

from .control_order_of_molecules import control_order_of_molecules
from .input_output import load_sdf
from .methods import Method
from .molecule import Molecule, create_molecule_from_charges



@jitclass({"molecules": ListType(Molecule.class_type.instance_type),
           "sdf_file": string,
           "ref_chg_file": string,
           "emp_chg_file": string,
           "num_of_molecules": int64,
           "num_of_atoms": int64,
           "ref_charges": float32[:],
           "emp_charges": float32[:],
           "emp_atomic_types_charges": DictType(string, float32[:]),
           "ref_atomic_types_charges": DictType(string, float32[:]),
           "all_atoms_id": int16[:],
           "parameters_per_atomic_type": int16,
           "atomic_types": ListType(string),
           "bonds_types": ListType(string)})
class SetOfMolecules:
    def __init__(self, molecules, sdf_file, atomic_types, bonds_types):
        self.molecules = molecules
        self.sdf_file = sdf_file
        self.num_of_molecules = len(molecules)
        self.num_of_atoms = sum(array([molecule.num_of_atoms for molecule in self.molecules], dtype=int64))
        self.atomic_types = atomic_types  # prepsat po fixu
        self.bonds_types = bonds_types  # prepsat po fixu



def create_set_of_molecules(sdf_file: str,
                            atomic_types_pattern: str) -> SetOfMolecules:

    print(f"Loading of set of molecules from {sdf_file}...")
    molecules_data = [x.splitlines() for x in open(sdf_file, "r").read().split("$$$$\n")][:-1]
    molecules = List([load_sdf(molecule_data, atomic_types_pattern)
                      for molecule_data in molecules_data])
    atomic_types = List(set([atom for molecule in molecules for atom in molecule.atoms_representation]))
    bonds_types = List(set([bond for molecule in molecules for bond in molecule.bonds_representation]))
    set_of_molecules = SetOfMolecules(molecules,
                                      sdf_file,
                                      atomic_types,
                                      bonds_types)
    print(f"    {set_of_molecules.num_of_molecules} molecules was loaded.")
    print(colored("ok\n", "green"))
    return set_of_molecules


def create_method_data(method: Method,
                       set_of_molecules: SetOfMolecules):

    for molecule in set_of_molecules.molecules:
        molecule.atoms_id = array([method.atomic_types.index(atomic_type)
                                   for atomic_type in molecule.atoms_representation],
                                  dtype=npint16) * len(method.parameters["atom"]["names"])
        molecule.distance_matrix = cdist(molecule.atomic_coordinates, molecule.atomic_coordinates).astype(npfloat32)
    if "bond" in method.parameters:
        for molecule in set_of_molecules.molecules:
            molecule.bonds_id = array([method.bond_types.index(bond)
                                       for bond in molecule.bonds_representation],
                                      dtype=npint16) + len(method.parameters["atom"]["names"]) * len(method.atomic_types)

    if "parameterization.py" in getmodule(stack()[1][0]).__file__:
        set_of_molecules.parameters_per_atomic_type = len(method.parameters["atom"]["names"])
        set_of_molecules.ref_charges = array([at_chg for molecule in set_of_molecules.molecules for at_chg in molecule.ref_charges], dtype=npfloat32)
        set_of_molecules.all_atoms_id = array([atom_id for molecule in set_of_molecules.molecules for atom_id in molecule.atoms_id], dtype=npint16)

        atomic_types_charges = defaultdict(list)
        for charge, symbol in zip(set_of_molecules.ref_charges, [symbol for molecule in set_of_molecules.molecules for symbol in molecule.atoms_representation]):
            atomic_types_charges[symbol].append(charge)

        set_of_molecules.ref_atomic_types_charges = Dict.empty(key_type=string, value_type=float32[:])
        for atomic_type, value in atomic_types_charges.items():
            set_of_molecules.ref_atomic_types_charges[atomic_type] = array(value, dtype=npfloat32)


def create_parameterization_validation_set(set_of_molecules, random_seed, parameterization_subset, method):
    print("Creating validation and parameterization sets...")
    random.seed(random_seed)
    set_of_molecules_parameterization = create_subset_of_molecules(set_of_molecules, method, parameterization_subset)
    if len(set_of_molecules_parameterization.molecules) == len(set_of_molecules.molecules):
        exit(colored("Error! It is too small set of molecules or too high parameterization_subset value.\n", "red"))
    parameterization_molecules_names = [molecule.name for molecule in set_of_molecules_parameterization.molecules]

    molecules_validation = List([molecule for molecule in set_of_molecules.molecules if molecule.name not in parameterization_molecules_names])

    atomic_types_validation = List(set([atom for molecule in molecules_validation for atom in molecule.atoms_representation]))
    bonds_types_validation = List(set([bond for molecule in molecules_validation for bond in molecule.bonds_representation]))

    set_of_molecules_validation = SetOfMolecules(molecules_validation, set_of_molecules.sdf_file, atomic_types_validation,
                                                 bonds_types_validation)
    print(f"    {set_of_molecules_parameterization.num_of_molecules} molecules in parameterization set.")
    print(f"    {set_of_molecules_validation.num_of_molecules} molecules in validation set.")
    print(colored("ok\n", "green"))
    return set_of_molecules_parameterization, set_of_molecules_validation


def create_subset_of_molecules(original_set_of_molecules, method, subset):
    def _select_molecules(all_molecules, subset, atomic_types):
        # create subset of molecules, which contain such molecules,
        # to contain at least subset_criterion atoms from all atomic types
        counter_atoms = Counter()
        molecules = []
        for molecule in all_molecules:
            atoms = molecule.atoms_representation
            if any(counter_atoms[atom] < subset for atom in atoms):
                counter_atoms.update(atoms)
                molecules.append(molecule)
            if all(counter_atoms[x] > subset for x in atomic_types):
                break
        return molecules

    molecules = _select_molecules(random.permutation(original_set_of_molecules.molecules), subset, method.atomic_types)
    molecules = _select_molecules(molecules[::-1], subset, method.atomic_types)
    if method == "ACKS2":
        # add such molecules to subset of molecules to contain all bond types
        bond_types = set([bond for molecule in original_set_of_molecules.molecules for bond in molecule.bonds_representation])
        counter_bonds = Counter()
        for molecule in molecules:
            counter_bonds.update(molecule.bonds_representation)

        if any(counter_bonds[bond] < subset // 5 + 1 for bond in bond_types):
            molecules_names = [molecule.name for molecule in molecules]
            for molecule in original_set_of_molecules.molecules:
                bonds = molecule.bonds_representation
                if any(counter_bonds[bond] < subset // 5 + 1 for bond in bonds) and molecule.name not in molecules_names:
                    counter_bonds.update(bonds)
                    molecules.append(molecule)
                    molecules_names.append(molecule.name)
                if all(counter_bonds[bond] >= subset // 5 + 1 for bond in bond_types):
                    break

    numba_molecules = List(molecules)


    atomic_types_n = List(set([atom for molecule in numba_molecules for atom in molecule.atoms_representation]))
    bonds_types_n = List(set([bond for molecule in numba_molecules for bond in molecule.bonds_representation]))


    subset_of_molecules = SetOfMolecules(numba_molecules, original_set_of_molecules.sdf_file, atomic_types_n, bonds_types_n)
    return subset_of_molecules


def create_set_of_molecules_from_chg_files(ref_chg_file, emp_chg_file):
    print(f"Loading of set of molecules from {ref_chg_file} and {emp_chg_file}...")
    ref_atomic_types_charges = defaultdict(list)
    emp_atomic_types_charges = defaultdict(list)
    molecules = List()
    all_ref_charges = []
    all_emp_charges = []
    with open(ref_chg_file, "r") as ref_chg_file, open(emp_chg_file, "r") as emp_chg_file:
        ref_molecules_data = [[line.split() for line in molecule.splitlines()]
                              for molecule in ref_chg_file.read().split("\n\n")[:-1]]
        emp_molecules_data = [[line.split() for line in molecule.splitlines()]
                              for molecule in emp_chg_file.read().split("\n\n")[:-1]]
        ref_molecules_names = [molecule[0][0] for molecule in ref_molecules_data]
        emp_molecules_names = [molecule[0][0] for molecule in emp_molecules_data]
        control_order_of_molecules(ref_molecules_names, emp_molecules_names, ref_chg_file.name, emp_chg_file.name)
        print("    Creation of molecules...")
        for name, ref_molecule, emp_molecule in zip(ref_molecules_names, ref_molecules_data, emp_molecules_data):
            atomic_symbols = List()
            [atomic_symbols.append(atom_line[1]) for atom_line in ref_molecule[2:]]
            ref_charges = array([float(atom_line[2]) for atom_line in ref_molecule[2:]], dtype=npfloat32)
            emp_charges = array([float(atom_line[2]) for atom_line in emp_molecule[2:]], dtype=npfloat32)
            all_ref_charges.append(ref_charges)
            all_emp_charges.append(emp_charges)
            molecules.append(create_molecule_from_charges(name, atomic_symbols, ref_charges, emp_charges))
            for symbol, ref_charge, emp_charge in zip(atomic_symbols, ref_charges, emp_charges):
                ref_atomic_types_charges[symbol].append(ref_charge)
                emp_atomic_types_charges[symbol].append(emp_charge)
    ref_atomic_types_charges_numba = Dict.empty(key_type=string, value_type=float32[:])
    emp_atomic_types_charges_numba = Dict.empty(key_type=string, value_type=float32[:])
    for (symbol, ref_charges), (_, emp_charges) in zip(sorted(ref_atomic_types_charges.items()), sorted(emp_atomic_types_charges.items())):
        ref_atomic_types_charges_numba[symbol] = array(ref_charges, dtype=npfloat32)
        emp_atomic_types_charges_numba[symbol] = array(emp_charges, dtype=npfloat32)
    print(colored("ok\n", "green"))


    atomic_types = List(set([atom for molecule in molecules for atom in molecule.atoms_representation]))
    bonds_types = List(set([bond for molecule in molecules for bond in molecule.bonds_representation]))


    set_of_molecules = SetOfMolecules(molecules, "", atomic_types, bonds_types)
    set_of_molecules.ref_chg_file = ref_chg_file.name
    set_of_molecules.emp_chg_file = emp_chg_file.name
    set_of_molecules.ref_atomic_types_charges = ref_atomic_types_charges_numba
    set_of_molecules.emp_atomic_types_charges = emp_atomic_types_charges_numba
    set_of_molecules.ref_charges = concatenate(all_ref_charges, axis=0)
    set_of_molecules.emp_charges = concatenate(all_emp_charges, axis=0)
    return set_of_molecules
