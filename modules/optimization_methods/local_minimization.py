from collections import namedtuple

from scipy.optimize import minimize
import numpy as np

from ..chg_methods.chg_method import ChargeMethod
from ..set_of_molecules import SetOfMolecules

def local_minimization(objective_function: "function",
                       set_of_mols_par: SetOfMolecules,
                       chg_method: ChargeMethod,
                       initial_params: np.array,
                       maxiter: int = 10000000) -> namedtuple:

    loc_min_course = []
    res = minimize(objective_function,
                   initial_params,
                   method="SLSQP",
                   options={"maxiter": maxiter, "ftol": 1e-10},
                   args=(chg_method, set_of_mols_par, loc_min_course))
    return namedtuple("chgs", ["params",
                               "obj_val",
                               "loc_min_courses"])(res.x,
                                                   res.fun,
                                                   [loc_min_course])
