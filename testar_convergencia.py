import os
import pickle
import random

from matplotlib import pyplot as plt
import numpy as np
from benchmark_functions import rastrigin_function as function
from benchmark_functions import adiciona_ruido

from rps import nelder_mead_base, nelder_mead_reset, rps_avg, rps_test, rps

path = "results/avals/"
if not os.path.isdir(path):
    os.makedirs(path)

desvio = 5

metodos = ["rps", "nelder_mead_base", "nelder_mead_reset", "rps_avg", "rps_test"]
labels = ["RPStau", "NMbas", "NMrst", "RPSavg", "RPStst"]
met_funs = [rps, nelder_mead_base, nelder_mead_reset, rps_avg, rps_test]
params = {}
for metodo in metodos:
    try:
        filename = 'results/params/params-' + str(desvio) + "/" + metodo + ".pkl"
        with open(filename, "rb") as fp:
            params[metodo] = pickle.load(fp)
    except:
        print("Parametros de " + metodo + " nao encontrados")
        params[metodo] = {}

opts = {"lu": [(-100, 100)], "qtd": 15, "dim": 20, "max_avals": 100000}

qtd = opts["qtd"]
dim = opts["dim"]
max_avals = opts["max_avals"]
lu = opts["lu"] * dim
x0s = [[(random.uniform(l, u)) for l, u in lu] for _ in range(qtd)]

amostras_mets = {metodo: [] for metodo in metodos}

ruido = adiciona_ruido(function, desvio=desvio)

for metodo, metodo_function in zip(metodos, met_funs):
    for x0 in x0s:
        x, best_sols = metodo_function(ruido, x0, lu, max_avals, **params[metodo], f_original = function)
        amostras_mets[metodo].append(best_sols)
    amostras_mets[metodo] = np.mean(amostras_mets[metodo], axis=0)

for (met, best_sols), label in zip(amostras_mets.items(), labels):
    plt.plot(range(len(best_sols)), np.log(np.array(best_sols) + 1), label=label)

plt.legend()
plt.xlabel("Avaliações")
plt.ylabel("Valor de objetivo")
plt.title(function.__name__)
filename = "results/avals/" + function.__name__
plt.savefig(filename + ".pdf")
plt.savefig(filename + ".png")