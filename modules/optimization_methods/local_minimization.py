from collections import namedtuple

from scipy.optimize import minimize
import numpy as np

from ..chg_methods.chg_method import ChargeMethod
from ..set_of_molecules import SetOfMolecules


def local_minimization(objective_function: "function",
                       set_of_mols: SetOfMolecules,
                       chg_method: ChargeMethod,
                       initial_params: np.array,
                       maxiter: int = 10000000) -> namedtuple:

    if maxiter == 0:
        return namedtuple("chgs", ["params",
                                   "obj_val",
                                   "loc_min_courses"])(initial_params,
                                                       1000,
                                                       [[1000]])


    loc_min_course = []
    res = minimize(objective_function,
                   initial_params,
                   # bounds = [[0.000001,5] for x in range(len(initial_params))], # POZOR!!!!!!!!!!!
                   method="SLSQP",
                   options={"maxiter": maxiter, "ftol": 1e-10, "eps":0.001},
                   args=(chg_method, set_of_mols, loc_min_course))
    return namedtuple("chgs", ["params",
                               "obj_val",
                               "loc_min_courses"])(res.x,
                                                   res.fun,
                                                   [loc_min_course])
