from datetime import date
from json import load, dumps
from sys import exit

import numpy as np
from termcolor import colored


class ChargeMethod:
    def __repr__(self):
        return self.__class__.__name__

    def load_params(self,
                    params_file: str,
                    ats_types_pattern: str = None) -> str:

        print(f"Loading of parameters from {params_file}...")
        if params_file:
            self.params = load(open(params_file))
            try:
                self.params["metadata"]["atomic_types_pattern"] = {"plain-ba": "ba", "plain-ba-ba": "ba2"}[self.params["metadata"]["atomic_types_pattern"]]
            except KeyError:
                pass

            self.ats_types_pattern = self.params["metadata"]["atomic_types_pattern"]
        else:
            self.params = self.params_pattern
            self.ats_types_pattern = ats_types_pattern
            self.params["metadata"]["atomic_types_pattern"] = ats_types_pattern

        self.params_per_at_type = len(self.params["atom"]["names"])
        method_in_params_file = self.params["metadata"]["method"]
        if self.__class__.__name__ != method_in_params_file:
            exit(colored(f"ERROR! These parameters are for method {method_in_params_file}, "
                         f"but you selected by argument --chg_method {self.__class__.__name__}!\n", "red"))
        print(colored("ok\n", "green"))
        return self.ats_types_pattern

    def prepare_params_for_calc(self,
                                set_of_molecules: "SetOfMolecules"):
        missing_at = set(set_of_molecules.ats_types) - set(self.params["atom"]["data"].keys())
        exit_status = False
        if missing_at:
            print(colored(f"ERROR! Atomic type(s) {', '.join(missing_at)} is not defined in parameters.", "red"))
            exit_status = True
        if "bond" in self.params:
            missing_bonds = set(set_of_molecules.bonds_types) - set(self.params["bond"]["data"].keys())
            if missing_bonds:
                print(colored(f"ERROR! Bond type(s) {', '.join(missing_bonds)} is not defined in parameters.", "red"))
                exit_status = True
        if exit_status:
            exit()
        self._dict_to_array()

    def prepare_params_for_par(self,
                               set_of_molecules: "SetOfMolecules"):
        missing_at = set(set_of_molecules.ats_types) - set(self.params["atom"]["data"].keys())
        for at in missing_at:
            for key, vals in self.params["atom"]["data"].items():
                if key.split("/")[0] == at.split("/")[0]:
                    self.params["atom"]["data"][at] = vals
                    print(colored(f"    Atom type {at} was added to parameters. "
                                  f"Parameters derived from {key}", "yellow"))
                    break
            else:
                self.params["atom"]["data"][at] = [np.random.random() for _ in range(len(self.params["atom"]["names"]))]
                print(colored(f"    Atom type {at} was added to parameters. Parameters are random numbers.", "yellow"))
        # unused atoms are atomic types which are in parameters but not in set of molecules
        unused_at = set(self.params["atom"]["data"].keys()) - set(set_of_molecules.ats_types)
        for at in unused_at:
            del self.params["atom"]["data"][at]
        if unused_at:
            print(colored(f"    {', '.join(unused_at)} was deleted from parameters, "
                          f"because of absence in set of molecules.", "yellow"))

        if "bond" in self.params:
            missing_bonds = set(set_of_molecules.bonds_types) - set(self.params["bond"]["data"].keys())
            for bond in missing_bonds:
                mat1, mat2, mtype = [value.split("/")[0] for value in bond.split("-")]
                for key, val in self.params["bond"]["data"].items():
                    pat1, pat2, ptype = [value.split("/")[0] for value in key.split("-")]
                    if (pat1, pat2, ptype) == (mat1, mat2, mtype):
                        self.params["bond"]["data"][bond] = val
                        print(colored(f"    Bond type {bond} was added to parameters. "
                                      f"Parameter derived from {key}", "yellow"))
                        break
                else:
                    self.params["bond"]["data"][bond] = np.random.random()
                    print(colored(f"    Bond type {bond} was added to parameters. "
                                  f"Parameter is random numbers.", "yellow"))
            # unused bonds are bond types which are in parameters but not in set of molecules
            unused_bonds = set(self.params["bond"]["data"].keys()) - set(set_of_molecules.bonds_types)
            for bond in unused_bonds:
                del self.params["bond"]["data"][bond]
            if unused_bonds:
                print(colored(f"    {', '.join(unused_bonds)} was deleted from parameters, "
                              f"because of absence in set of molecules", "yellow"))
        self._dict_to_array()

    def _dict_to_array(self):
        self.ats_types = sorted(self.params["atom"]["data"].keys())
        if "bond" in self.params:
            self.bond_types = sorted(self.params["bond"]["data"].keys())
        params_vals = []
        for _, vals in sorted(self.params["atom"]["data"].items()):
            params_vals.extend(vals)
        if "bond" in self.params:
            for _, val in sorted(self.params["bond"]["data"].items()):
                params_vals.append(val)
        if "common" in self.params:
            params_vals.extend(self.params["common"].values())

        params_vals = [round(x, 4) for x in params_vals]

        self.params_vals = np.array(params_vals, dtype=np.float64)
        self.bounds = (min(self.params_vals), max(self.params_vals))

    def new_params(self,
                   new_params: np.array,
                   sdf_file: str,
                   new_params_file: str,
                   original_params_file: str,
                   ref_chg_file: str,
                   date: date):

        print(f"Writing parameters to {new_params_file}...")
        self.params_vals = new_params
        params_per_at = len(self.params["atom"]["names"])
        index = 0
        for at in self.ats_types:
            self.params["atom"]["data"][at] = list(self.params_vals[index: index + params_per_at])
            index = index + params_per_at

        if "bond" in self.params:
            for bond in self.bond_types:
                self.params["bond"]["data"][bond] = self.params_vals[index]
                index += 1

        if "common" in self.params:
            for common in sorted(self.params["common"]):
                self.params["common"][common] = self.params_vals[index]
                index += 1

        self.params["metadata"]["sdf file"] = sdf_file
        self.params["metadata"]["reference charges file"] = ref_chg_file
        self.params["metadata"]["original parameters file"] = original_params_file
        self.params["metadata"]["date"] = date

        with open(new_params_file, "w") as new_params_file:
            new_params_file.write(dumps(self.params, indent=2, sort_keys=True))
        print(colored("ok\n", "green"))
