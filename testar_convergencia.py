import os
import pickle
import random
import sys

from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

import benchmark_functions as bf
from rps import nelder_mead_base, nelder_mead_reset, rps_avg, rps_test, rps

path_params = ''
path_results = ''
if len(sys.argv) >= 2:
    path_params = 'results/params/' + sys.argv[1] + '/'
    path_results = 'results/out/' + sys.argv[1] + '/'
    if not os.path.isdir(path_params):
        os.makedirs(path_params)
    if not os.path.isdir(path_results):
        os.makedirs(path_results)

def get_params(alg, f, d, r):
    params = {}
    try:
        filename = path_params + alg + "-" + f + "-" + str(d) + "-" + str(r)
        with open(filename + "-par.pkl", "rb") as fp:
            params = pickle.load(fp)
    except:
        print("Parametros de " + filename + " nao encontrados")
    
    return params

ruidos = [1, 5]
dims = [10]
functions = [
    bf.zakharov_function,
    bf.rastrigin_function
    ]
metodos = [rps, nelder_mead_base, nelder_mead_reset, rps_avg, rps_test]
algoritmos = ["rps", "nelder_mead_base", "nelder_mead_reset", "rps_avg", "rps_test"]
labels = ["RPStau", "NMbas", "NMrst", "RPSavg", "RPStst"]

# params[alg][f_name_][d][r]
params = {
    alg: {
        f.__name__: {
            d: {
                r: get_params(alg, f.__name__, d, r)
                for r in ruidos
            }
            for d in dims
        }
        for f in functions
    }
    for alg in algoritmos
}

opts = {"lu": [(-100, 100)], "qtd": 5, "max_avals": 30}

qtd = opts["qtd"]
max = opts["max_avals"]
lu = opts["lu"]
x0s = [[(random.uniform(l, u)) for l, u in lu] for _ in range(qtd)]

final_sols = {alg: [] for alg in algoritmos}


for fun in functions:
    for ruido in ruidos:
        funruido = bf.adiciona_ruido(fun, desvio=ruido)
        for dim in dims:
            bounds = lu*dim
            max_avals = max*dim
            amostras_mets = {alg: [] for alg in algoritmos}
            for metodo, alg in zip(metodos, algoritmos):
                parametros = params[alg][fun.__name__][dim][ruido]
                for x0 in x0s:
                    x, best_sols = metodo(funruido, x0, lu, max_avals, **parametros, f_original = fun)
                    amostras_mets[alg].append(best_sols)
                    final_sols[alg].append(best_sols[-1])
                amostras_mets[alg] = np.mean(amostras_mets[alg], axis=0)
            for alg, label in zip(algoritmos, labels):
                plt.plot(range(len(amostras_mets[alg])), np.log(np.array(amostras_mets[alg]) + 1), label=label)
            plt.legend()
            plt.xlabel("Avaliações")
            plt.ylabel("Valor de objetivo")
            plt.title(fun.__name__)
            filename = path_results + fun.__name__ + "-" + str(dim) + "-" + str(ruido)
            plt.savefig(filename + ".pdf")
            plt.savefig(filename + ".png")
            plt.clf()

data = [np.log(np.array(final_sols[alg]) + 1) for alg in algoritmos]
#data = [np.array(final_sols[alg])for alg in algoritmos]

data = np.array(data)
np.save(path_results + "data.npy", data)

df = pd.DataFrame(data.T, columns=labels)

ax = sns.violinplot(data=df, palette=sns.color_palette(), cut=0, inner=None)
for item in ax.collections:
    x0, y0, width, height = item.get_paths()[0].get_extents().bounds
    item.set_clip_path( plt.Rectangle((x0, y0), width/2, height, transform=ax.transData))
num_items = len(ax.collections)
sns.stripplot(data=df, palette=sns.color_palette(), alpha=0.4, size=7)
for item in ax.collections[num_items:]:
    item.set_offsets(item.get_offsets() + 0.15)
sns.boxplot(data=df, width=0.25, showfliers=False, showmeans=True, meanprops=dict(marker='o', markerfacecolor='darkorange', markersize=10, zorder=3),
boxprops=dict(facecolor=(0,0,0,0), linewidth=3, zorder=3), whiskerprops=dict(linewidth=3), capprops=dict(linewidth=3), medianprops=dict(linewidth=3))
plt.legend(frameon=False, fontsize=15, loc='lower left')
# plt.title('', fontsize=28)
plt.title('Violin Plot')
plt.xlabel('Algoritmos')
plt.ylabel('Valor de função')
# plt.xticks(fontsize=15)
# plt.yticks(fontsize=15)
# sns.despine()
plt.savefig(path_results + 'violin_plot.pdf')
plt.savefig(path_results + 'violin_plot.png')
