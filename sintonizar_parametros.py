import os
import pickle
import random
import sys

import numpy as np

from matplotlib import pyplot as plt
plt.rcParams['text.usetex'] = True

from skopt.plots import plot_convergence
from skopt.space import Real, Integer, Categorical
from skopt.utils import use_named_args
from skopt import gp_minimize

import benchmark_functions as bf
from rps import nelder_mead_base, nelder_mead_reset, rps_avg, rps_test, rps
from gsa import ga, cmaes, pso

np.int = np.int_

path = ''
if len(sys.argv) >= 2:
    path = 'results/params/' + sys.argv[1] + '/'
    if not os.path.isdir(path):
        os.makedirs(path)

space_rps = [
    Real(1.1, 3, name="dr"),
    Real(1.1, 3, name="de"),
    Real(0.1, 0.9, name="dc"),
    Real(0.1, 0.9, name="ds"),
    Real(0, 1e-03, name="eps_x"),
    Integer(1, 10, name="emax"),
    Real(0, 3, name="tau"),
    Real(0.01, 0.10, name="alpha"),
    Categorical(["ttest", "wilcoxon", "none"], name="teste")
]

space_nelder_mead_base = [
    Real(1.1, 3, name="dr"),
    Real(1.1, 3, name="de"),
    Real(0.1, 0.9, name="dc"),
    Real(0.1, 0.9, name="ds")
]

space_nelder_mead_reset = [
    Real(1.1, 3, name="dr"),
    Real(1.1, 3, name="de"),
    Real(0.1, 0.9, name="dc"),
    Real(0.1, 0.9, name="ds"),
    Real(0, 1e-03, name="eps_x"),
]

space_rps_avg = [
    Real(1.1, 3, name="dr"),
    Real(1.1, 3, name="de"),
    Real(0.1, 0.9, name="dc"),
    Real(0.1, 0.9, name="ds"),
    Real(0, 1e-03, name="eps_x"),
    Integer(1, 10, name="emax"),
]

space_rps_test = [
    Real(1.1, 3, name="dr"),
    Real(1.1, 3, name="de"),
    Real(0.1, 0.9, name="dc"),
    Real(0.1, 0.9, name="ds"),
    Real(0, 1e-03, name="eps_x"),
    Integer(1, 10, name="emax"),
    Real(0.01, 0.10, name="alpha"),
    Categorical(["ttest", "wilcoxon", "none"], name="teste")
]

space_ga = [
    Integer(5, 100, name="pop_size"),
    Real(0.5, 1.0, name="prob_sbx"),
    Real(0.5, 1.0, name="prob_var_sbx"),
    Real(0.5, 1.5, name="eta_sbx"),
    Real(0.1, 1.0, name="prob_mut"),
    Real(0.1, 1.0, name="eta_mut")
]

space_cmaes = [
    Real(0.01, 2, name='sigma'),
    Categorical([False, True], name='normalize'),
    Integer(0, 10, name='restarts'),
    Categorical([False, True], name='restart_from_best'),
    Integer(1, 10, name='incpopsize'),
    Categorical([False, True], name='eval_initial_x'),
    Real(0.5, 2, name='noise_change_sigma_exponent'),
    Real(0, 0.5, name='noise_kappa_exponent'),
    Categorical([False, True], name='bipop')
]

space_pso = [
    Integer(5, 100, name='pop_size'),
    Real(0.5, 1.5, name='w'),
    Real(0.5, 3, name='c1'),
    Real(0.5, 3, name='c2'),
    Categorical([False, True], name='adaptive'),
    Real(0.05, 2, name='max_velocity_rate'),
    Categorical([False, True], name='pertube_best')
]

ruidos = [1, 2, 5]
dims = [10, 20, 30]
functions = [
    bf.zakharov_function,
    bf.rosenbrock_function,
    bf.expanded_schaffer_function,
    bf.rastrigin_function,
    bf.levy_function,
    bf.bent_cigar_function,
    bf.hgbat_function,
    bf.high_conditioned_elliptic_function,
    bf.katsuura_function,
    bf.happycat_function,
    bf.expanded_rosenbrocks_plus_griewangk_function,
    bf.modified_schwefels_function,
    bf.ackleys_function,
    bf.discus_function,
    bf.griewanks_function,
    bf.schaffer_f7_function,
    ]
function_labels = [
    "Zakharov",
    "Rosenbrock",
    "Schaffer",
    "Rastrigin",
    "Levy",
    "Bent Cigar",
    "HGBat",
    "Elliptic",
    "Katsuura",
    "HappyCat",
    "Rosenbrock+Griewangk",
    "Schwefel",
    "Ackley",
    "Discus",
    "Griewank",
    "Schaffer",
    ]
metodos = [rps, nelder_mead_reset, ga, cmaes, pso]
algoritmos = ["rps", "nelder_mead_reset", "ga", "cmaes", "pso"]
spaces = [space_rps, space_nelder_mead_reset, space_ga, space_cmaes, space_pso]


for fun, fun_label in zip(functions, function_labels):
    for ruido in ruidos:
        funruido = bf.adiciona_ruido(fun, desvio=ruido)
        for dim in dims:
            qtd = 5
            bounds = [(-50, 50)]*dim
            max_avals = 100*dim
            x0s = [[(random.uniform(l, u)) for l, u in bounds] for _ in range(qtd)]
            for space, metodo, alg in zip(spaces, metodos, algoritmos):
                
                @use_named_args(space)
                def eval(**params):
                    # print(params)
                    return np.median([
                        fun(metodo(f=funruido, x0=x0, max_avals=max_avals, lu=bounds, **params).X)
                        for x0 in x0s
                        ])
                
                res_gp = gp_minimize(eval, space, n_calls=30) #, random_state=0
                params_sintonizados = {s.name: p for s, p in zip(space, res_gp.x)}
                filename = path + alg + "-" + fun.__name__ + "-" + str(dim) + "-" + str(ruido)
                with open(filename + "-par.pkl", "wb") as fp:
                    pickle.dump(params_sintonizados, fp)
                ax = plot_convergence(res_gp)
                plt.xlabel('Number of evaluations')
                plt.ylabel('Utility function value')
                plt.title(fun_label + ': Convergence plot for ' + " $\sigma = $" + str(ruido))
                ax.figure.savefig(filename + "-conv_plot.png")
                ax.figure.savefig(filename + "-conv_plot.pdf")
                ax.figure.clear()
