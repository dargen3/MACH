import numpy as np

from .chg_methods.chg_method import ChargeMethod
from .set_of_molecules import SetOfMolecules


class ObjectiveFunction:

    def __init__(self):
        self.obj_eval = 0
        self.obj_eval_error = 0


    def calculate(self,
                  params: np.array,
                  chg_method: ChargeMethod,
                  set_of_mols: SetOfMolecules,
                  obj_vals: list = None) -> float:

        def _rmsd_calculation(emp_chgs: np.array) -> tuple:
            ats_types_rmsd = np.empty(len(chg_method.ats_types))
            for index, symbol in enumerate(chg_method.ats_types):
                differences = emp_chgs[set_of_mols.all_ats_ids == index * chg_method.params_per_at_type] - set_of_mols.ref_ats_types_chgs[symbol]
                ats_types_rmsd[index] = np.sqrt(np.mean(np.abs(differences) ** 2))
            total_mols_rmsd = 0
            index = 0
            for mol in set_of_mols.mols:
                new_index = index + mol.num_of_ats
                differences = emp_chgs[index: new_index] - mol.ref_chgs
                total_mols_rmsd += np.sqrt(np.mean(np.abs(differences) ** 2))
                index = new_index
            return ats_types_rmsd, total_mols_rmsd / set_of_mols.num_of_mols

        self.obj_eval += 1
        chg_method.params_vals = params

        try:
            emp_chgs = chg_method.calculate(set_of_mols)
        except (np.linalg.LinAlgError, ZeroDivisionError) as e:
            self.obj_eval_error += 1
            print(e)
            return 1000.0
        ats_types_rmsd, rmsd = _rmsd_calculation(emp_chgs)
        objective_val = rmsd + np.mean(ats_types_rmsd)
        if np.isnan(objective_val):
            return 1000.0
        if isinstance(obj_vals, list):
            obj_vals.append(objective_val)
        print("    Average molecules RMSD: {}    Worst at. type RMSD: {}     {}".format(str(rmsd)[:8], str(np.max(ats_types_rmsd))[:8], objective_val), end="\r")  # f forma předělat!
        return objective_val
