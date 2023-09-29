import os
import pickle
import random
import sys

import numpy as np
from scipy import optimize

import benchmark_functions as bf
import rps
import nelder_mead
import rps_avg
import rps_tt

import analise_resultado

files = ["rps.pkl", "nelder_mead.pkl", "rps_avg.pkl", "rps_tt.pkl"]
metodos = ["rps", "nelder_mead", "rps_avg", "rps_tt"]
params = {}
for file, metodo in zip(files, metodos):
    try:
        with open(file, "rb") as fp:
            params[metodo] = pickle.load(fp)
    except:
        params[metodo] = {}

def media(f, x):
    return np.mean([f(x) for _ in range(100)])

def string_point(x):
    s = "("
    for xi in x[:-1]:
        s += f"{xi:.3f}, "
    s += f"{x[-1]:.3f})"
    return s

def criar_ponto(lu):
    x = []
    for i in range(len(lu)):
        x.append(random.uniform(lu[i][0], lu[i][1]))
    return x

def test_function(f, x0, lu, max_avals, f_name=None):
    p_rps = rps.rps(f, x0, lu, max_avals,
                 **params["rps"]
                 ).x
    p_nelder_mead = nelder_mead.nelder_mead(f, x0, lu, max_avals,
                 **params["nelder_mead"]
                 ).x
    p_rps_avg = rps_avg.rps_avg(f, x0, lu, max_avals,
                 **params["rps_avg"]
                 ).x
    p_rps_tt = rps_tt.rps_tt(f, x0, lu, max_avals,
                 **params["rps_tt"]
                 ).x
    
    return (p_rps, p_nelder_mead, p_rps_avg, p_rps_tt)

path = ''

desvio = 10

if len(sys.argv) >= 2:
    path = 'results/' + sys.argv[1] + '/'
    if not os.path.isdir(path):
        os.makedirs(path)

if len(sys.argv) >= 3:
    desvio = float(sys.argv[2])

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

opts = {"lu": [(-5, 5)], "qtd": 30, "dim": 10, "max_avals": 1000}
#opts = {"lu": [(-5, 5)], "qtd": 10, "dim": 5, "max_avals": 500}
#opts = {"lu": [(-5, 5)], "qtd": 2, "dim": 2, "max_avals": 200}


lu = opts["lu"]

rpsMedias = []
nelder_meadMedias = []
rps_avgMedias = []
rps_ttMedias = []

for function in functions:
    ruido = bf.adiciona_ruido(function, desvio=desvio)
    qtd = opts["qtd"]
    dim = opts["dim"]
    max_avals = opts["max_avals"]
    rpsMedia, nelder_meadMedia, rps_avgMedia, rps_ttMedia = 0, 0, 0, 0
    for i in range(qtd):
        limites_lu = lu*dim
        rpsP, nmP, ravgP, rttP = test_function(ruido, criar_ponto(limites_lu), limites_lu, max_avals, function.__name__)
        rpsMedia += function(rpsP)
        nelder_meadMedia += function(nmP)
        rps_avgMedia += function(ravgP)
        rps_ttMedia += function(rttP)
    rpsMedias.append(rpsMedia / qtd)
    nelder_meadMedias.append(nelder_meadMedia / qtd)
    rps_avgMedias.append(rps_avgMedia / qtd)
    rps_ttMedias.append(rps_ttMedia / qtd)

data = [
    rpsMedias,
    nelder_meadMedias,
    rps_avgMedias,
    rps_ttMedias
]

data = np.array(data)
np.save(path + "data.npy", data)
analise_resultado.analisar_resultado(data, path, "Nível de ruído: " + str(desvio))
