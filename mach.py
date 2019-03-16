#!/usr/bin/env python3
from termcolor import colored
from numpy import random
import warnings
warnings.filterwarnings("ignore")
from modules.arguments import load_arguments
from modules.set_of_molecules import SetOfMolecules
from modules.calculation import Calculation
from modules.comparison import Comparison
from modules.parameterization import Parameterization
from modules.calculation_meta import calculation_meta  # only for my usage
from modules.parameterization_meta import parameterization_meta # only for my usage
from modules.make_complete_html import make_complete_html
from modules.clusterization import clusterize
from numba import jit
import nlopt


@jit(nopython=True, cache=True)
def numba_seed(seed):
    random.seed(seed)


if __name__ == '__main__':
    args = load_arguments()
    if args.random_seed != 0:
        random.seed(args.random_seed)
        numba_seed(args.random_seed)
        nlopt.srand(args.random_seed)
    print(colored("\nMACH is running with mode: {}\n".format(args.mode), "blue"))
    if args.mode == "set_of_molecules_info":
        set_of_molecules = SetOfMolecules(args.sdf, args.num_of_molecules)
        set_of_molecules.info(args.atomic_types_pattern)

    elif args.mode == "calculation":
        Calculation(args.sdf,
                    args.method,
                    args.parameters,
                    args.charges,
                    args.atomic_types_pattern,
                    args.rewriting_with_force)

    elif args.mode == "parameterization":
        Parameterization(args.sdf,
                         args.ref_charges,
                         args.parameters,
                         args.method,
                         args.optimization_method,
                         args.minimization_method,
                         args.atomic_types_pattern,
                         args.num_of_molecules,
                         args.num_of_samples,
                         args.subset_heuristic,
                         args.validation,
                         args.cpu,
                         args.data_dir,
                         args.rewriting_with_force,
                         args.git_hash)

    elif args.mode == "comparison":
        Comparison(args.ref_charges,
                   args.charges,
                   args.data_dir,
                   args.rewriting_with_force)

    elif args.mode == "calculation_meta":  # only for my usage
        calculation_meta(args.sdf,
                         args.method,
                         args.parameters,
                         args.charges,
                         args.atomic_types_pattern,
                         args.RAM,
                         args.walltime)

    elif args.mode == "parameterization_meta":  # only for my usage
        parameterization_meta(args.sdf,
                              args.ref_charges,
                              args.parameters,
                              args.method,
                              args.optimization_method,
                              args.minimization_method,
                              args.atomic_types_pattern,
                              args.num_of_molecules,
                              args.num_of_samples,
                              args.subset_heuristic,
                              args.validation,
                              args.cpu,
                              args.RAM,
                              args.walltime)

    elif args.mode == "make_complete_html":
        make_complete_html()

    elif args.mode == "clusterization":
        clusterize(args.charges, args.sdf)

    elif args.mode == "neural_network":
        with open(args.ref_charges, "r") as right_charges_file:
            list_with_right_charges = []
            for line in right_charges_file:
                ls = line.split()
                if len(ls) not in {0, 1, 3}:
                    exit(colored("File " + file + " with right charges is wrong! \n", "red"))
                if len(ls) == 3:
                    list_with_right_charges.append(float(ls[2]))
        set_of_molecules = SetOfMolecules(args.sdf)
        electronegativity = {"C":2.55, "H":2.2, "O":3.44, "N":3.04, "S":2.58}
        atoms_data = []
        from scipy import spatial
        for molecule in set_of_molecules:
            distances = spatial.distance.cdist(molecule.atomic_coordinates, molecule.atomic_coordinates)
            for index, atom in enumerate(molecule.atoms):
                atom_data = [electronegativity[atom.plain]]
                el = 0
                for i, a in enumerate(molecule.atoms):
                    if i == index:
                        continue
                    el += ((electronegativity[atom.plain]-electronegativity[a.plain])/2)/(distances[i][index])**3
                atom_data.append(el/(len(distances)-1))
                atom_data.append(atom.hbo.split("~")[1])
                atoms_data.append(atom_data)

        # from pprint import pprint ; pprint(atoms_data)
        # from sys import exit ; exit()


        from keras.models import Sequential
        from keras.layers import Dense
        from numpy import array, sqrt, mean, abs
        X = array(atoms_data)
        Y = array(list_with_right_charges)

        # model = Sequential()
        #
        # a = "tanh"
        #
        # model.add(Dense(10, activation=a, input_dim=3, init='uniform'))
        # model.add(Dense(10, activation=a, input_dim=10, init='uniform'))
        # model.add(Dense(10, activation=a, input_dim=10, init='uniform'))
        # model.add(Dense(output_dim=1, activation=a, input_dim=10))
        # model.compile(loss='mean_absolute_error', optimizer="adam")
        # model.fit(X, Y, epochs=200, batch_size=10000)
        # scores = model.evaluate(X, Y)
        #

        import pandas
        from keras.models import Sequential
        from keras.layers import Dense
        from keras.wrappers.scikit_learn import KerasRegressor
        from sklearn.model_selection import cross_val_score
        from sklearn.model_selection import KFold
        from sklearn.preprocessing import StandardScaler
        from sklearn.pipeline import Pipeline

        def baseline_model():
            # create model
            model = Sequential()
            model.add(Dense(10, input_dim=3, kernel_initializer='normal', activation='sigmoid'))
            model.add(Dense(10, input_dim=10, kernel_initializer='normal', activation='sigmoid'))
            model.add(Dense(1, kernel_initializer='normal', input_dim=10))
            # Compile model
            model.compile(loss='mean_absolute_error', optimizer='adam')
            return model


        estimator = KerasRegressor(build_fn=baseline_model, epochs=200, batch_size=10)
        estimator.fit(X, Y)
        pY = estimator.predict(X)








        from matplotlib import pyplot as plt

        print(sqrt(mean(abs(pY - Y)**2)))
        plt.plot(pY, Y, "*")
        plt.show()


