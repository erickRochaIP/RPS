import pickle
import random

import numpy as np
from skopt.space import Real, Integer, Categorical
from skopt.utils import use_named_args
from skopt import gp_minimize

import benchmark_functions as bf
from rps import rps
from nelder_mead import nelder_mead
from rps_avg import rps_avg
from rps_tt import rps_tt

np.int = np.int_

space_rps = [
    Real(1.1, 3, name="dr"),
    Real(1.1, 3, name="de"),
    Real(0.1, 0.9, name="dc"),
    Real(0.1, 0.9, name="ds"),
    Real(0, 1e-03, name="eps_x"),
    Integer(1, 10, name="emax"),
    Real(0, 3, name="tau"),
    Real(0.01, 0.10, name="alpha"),
    Categorical(["ttest", "wilcoxon"], name="teste")
]

space_nelder_mead = [
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

space_rps_tt = [
    Real(1.1, 3, name="dr"),
    Real(1.1, 3, name="de"),
    Real(0.1, 0.9, name="dc"),
    Real(0.1, 0.9, name="ds"),
    Real(0, 1e-03, name="eps_x"),
    Integer(1, 10, name="emax"),
    Real(0.01, 0.10, name="alpha"),
    Categorical(["ttest", "wilcoxon"], name="teste")
]

qtd = 1
num_var = 10
bounds = [(-10, 10)]*num_var
max_avals = 200
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

functions_com_ruido = [bf.adiciona_ruido(f, desvio=2) for f in functions]

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

@use_named_args(space_nelder_mead)
def eval_nelder_mead(**params):
    print(params)
    return media(params, nelder_mead)

@use_named_args(space_rps_avg)
def eval_rps_avg(**params):
    print(params)
    return media(params, rps_avg)

@use_named_args(space_rps_tt)
def eval_rps_tt(**params):
    print(params)
    return media(params, rps_tt)

evals = [eval_rps, eval_nelder_mead, eval_rps_avg, eval_rps_tt]
spaces = [space_rps, space_nelder_mead, space_rps_avg, space_rps_tt]
files = ["rps.pkl", "nelder_mead.pkl", "rps_avg.pkl", "rps_tt.pkl"]

for eval, space, file in zip(evals, spaces, files):
    res_gp = gp_minimize(eval, space, n_calls=50, random_state=0)
    params_sintonizados = {s.name: p for s, p in zip(space, res_gp.x)}
    with open(file, 'wb') as fp:
        pickle.dump(params_sintonizados, fp)