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
desvio = 10
if len(sys.argv) >= 2:
    path = 'results/params/' + sys.argv[1] + '/'
    if not os.path.isdir(path):
        os.makedirs(path)

if len(sys.argv) >= 3:
    desvio = float(sys.argv[2])

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

qtd = 1
num_var = 5
bounds = [(-10, 10)]*num_var
max_avals = 800
x0s = [[(random.uniform(l, u)) for l, u in bounds] for _ in range(qtd)]
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

functions_com_ruido = [bf.adiciona_ruido(f, desvio=desvio) for f in functions]

def media(params, metodo):
    media_funcoes = 0
    for fun, funruido in zip(functions, functions_com_ruido):
        media_x0s = 0
        for x0 in x0s:
            media_x0s += fun(
                            metodo(f=funruido, x0=x0, max_avals=max_avals, lu=bounds, **params).x
                        )
            
        media_funcoes += media_x0s / len(x0s)
        
    return media_funcoes / len(functions)
    

@use_named_args(space_rps)
def eval_rps(**params):
    print(params)
    return media(params, rps)

@use_named_args(space_nelder_mead_base)
def eval_nelder_mead_base(**params):
    print(params)
    return media(params, nelder_mead_base)

@use_named_args(space_nelder_mead_reset)
def eval_nelder_mead_reset(**params):
    print(params)
    return media(params, nelder_mead_reset)

@use_named_args(space_rps_avg)
def eval_rps_avg(**params):
    print(params)
    return media(params, rps_avg)

@use_named_args(space_rps_test)
def eval_rps_test(**params):
    print(params)
    return media(params, rps_test)

evals = [eval_rps, eval_nelder_mead_base, eval_nelder_mead_reset, eval_rps_avg, eval_rps_test]
spaces = [space_rps, space_nelder_mead_base, space_nelder_mead_reset, space_rps_avg, space_rps_test]
files = ["rps", "nelder_mead_base", "nelder_mead_reset", "rps_avg", "rps_test"]

for eval, space, file in zip(evals, spaces, files):
    res_gp = gp_minimize(eval, space, n_calls=20, random_state=0)
    params_sintonizados = {s.name: p for s, p in zip(space, res_gp.x)}
    with open(path + file + ".pkl", 'wb') as fp:
        pickle.dump(params_sintonizados, fp)
    ax = plot_convergence(res_gp)
    ax.figure.savefig(path + "conv_plot_" + file + ".png")
    ax.figure.clear()