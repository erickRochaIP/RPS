import os
import pickle
import random
import sys

import numpy as np
from skopt.plots import plot_convergence
from skopt.space import Real, Integer, Categorical
from skopt.utils import use_named_args
from skopt import gp_minimize

import benchmark_functions as bf
from rps import nelder_mead_base, nelder_mead_reset, rps_avg, rps_test, rps

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

ruidos = [1, 5]
dims = [10]
functions = [
    bf.zakharov_function,
    bf.rastrigin_function
    ]
metodos = [rps, nelder_mead_base, nelder_mead_reset, rps_avg, rps_test]
algoritmos = ["rps", "nelder_mead_base", "nelder_mead_reset", "rps_avg", "rps_test"]
spaces = [space_rps, space_nelder_mead_base, space_nelder_mead_reset, space_rps_avg, space_rps_test]


for fun in functions:
    for ruido in ruidos:
        funruido = bf.adiciona_ruido(fun, desvio=ruido)
        for dim in dims:
            qtd = 1
            bounds = [(-100, 100)]*dim
            max_avals = 10*dim
            x0s = [[(random.uniform(l, u)) for l, u in bounds] for _ in range(qtd)]
            for space, metodo, alg in zip(spaces, metodos, algoritmos):
                
                @use_named_args(space)
                def eval(**params):
                    print(params)
                    return sum([
                        fun(metodo(f=funruido, x0=x0, max_avals=max_avals, lu=bounds, **params).x)
                        for x0 in x0s
                        ])
                
                res_gp = gp_minimize(eval, space, n_calls=30, random_state=0)
                params_sintonizados = {s.name: p for s, p in zip(space, res_gp.x)}
                filename = path + alg + "-" + fun.__name__ + "-" + str(dim) + "-" + str(ruido)
                with open(filename + "-par.pkl", "wb") as fp:
                    pickle.dump(params_sintonizados, fp)
                ax = plot_convergence(res_gp)
                ax.figure.savefig(filename + "-conv_plot.png")
                ax.figure.clear()
