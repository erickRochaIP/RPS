import random

import numpy as np
from skopt.space import Real, Integer
from skopt.utils import use_named_args
from skopt import gp_minimize

import benchmark_functions as bf
from rps import rps

np.int = np.int_

space = [
    Real(1.1, 3, name="dr"),
    Real(1.1, 3, name="de"),
    Real(0.1, 0.9, name="dc"),
    Real(0.1, 0.9, name="ds"),
    Real(0, 3, name="crescimento"), #modificar o nome para "tau"
    Real(0, 1e-03, name="eps_x"),
    Integer(1, 10, name="emax") #modoficado para ser possível comparar com o Nelder Mead, valor anterior [2,10]
]

qtd = 5
num_var = 10 #num_var modificado, valor anterior 5
bounds = [(-10, 10)]*num_var #bounds modificado, valor anterior [-5,5]
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


@use_named_args(space)
def eval_rps(**params):
    print(params)
    media_funcoes = 0
    for fun, funruido in zip(functions, functions_com_ruido):
        media_x0s = 0
        for x0 in x0s:
            media_x0s += fun(
                            rps(f=funruido, x0=x0, max_avals=max_avals, lu=bounds, **params).x
                        )
            
        media_funcoes += media_x0s / len(x0s)
        
    return media_funcoes / len(functions)

res_gp = gp_minimize(eval_rps, space, n_calls=50, random_state=0) #n_calls modificado, valor anterior 15
print(res_gp.x)
