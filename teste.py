import os
import pickle
import random
import sys

import numpy as np
from scipy import optimize

import benchmark_functions as bf
from rps import nelder_mead_base, nelder_mead_reset, rps_avg, rps_test, rps

import analise_resultado

path = ''
desvio = 10

if len(sys.argv) >= 2:
    path = 'results/' + sys.argv[1] + '/'
    if not os.path.isdir(path):
        os.makedirs(path)

if len(sys.argv) >= 3:
    desvio = float(sys.argv[2])

metodos = ["rps", "nelder_mead_base", "nelder_mead_reset", "rps_avg", "rps_test"]
met_funs = [rps, nelder_mead_base, nelder_mead_reset, rps_avg, rps_test]
params = {}
for metodo in metodos:
    try:
        filename = 'results/params/params-' + sys.argv[2] + "/" + metodo + ".pkl"
        with open(filename, "rb") as fp:
            params[metodo] = pickle.load(fp)
    except:
        params[metodo] = {}

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

opts = {"lu": [(-5, 5)], "qtd": 30, "dim": 10, "max_avals": 1000}
#opts = {"lu": [(-5, 5)], "qtd": 10, "dim": 5, "max_avals": 500}
#opts = {"lu": [(-5, 5)], "qtd": 2, "dim": 2, "max_avals": 200}

qtd = opts["qtd"]
dim = opts["dim"]
max_avals = opts["max_avals"]
lu = opts["lu"] * dim
x0s = [[(random.uniform(l, u)) for l, u in lu] for _ in range(qtd)]

amostras_mets = {metodo: {} for metodo in metodos}

for metodo, metodo_function in zip(metodos, met_funs):
    for function, ruido in zip(functions, functions_com_ruido):
        amostras_mets[metodo][function.__name__] = []
        for x0 in x0s:
            amostras_mets[metodo][function.__name__].append(
                function(
                    metodo_function(ruido, x0, lu, max_avals, **params[metodo]).x
                    )
            )

data = [
    [
        amostras_mets[metodo][function_name]
        for function_name in amostras_mets[metodo]
    ]
    for metodo in amostras_mets
]

data = np.array(data)
np.save(path + "data.npy", data)
analise_resultado.analisar_resultado(data, path, "Nível de ruído: " + str(desvio))
