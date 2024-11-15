import os
import pickle
import random
import sys

from matplotlib import pyplot as plt
plt.rcParams['text.usetex'] = True

import numpy as np
import pandas as pd
import seaborn as sns

import benchmark_functions as bf
from rps import nelder_mead_base, nelder_mead_reset, rps_avg, rps_test, rps
from gsa import ga, cmaes, pso

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

def gerar_grafico_evolucao(amostras_mets, fun, fun_label, dim, ruido):
    for alg, label in zip(algoritmos, labels):
        plt.plot(range(len(amostras_mets[alg])), np.log(np.array(amostras_mets[alg]) + 1), label=label)
    plt.legend()
    plt.xlabel("Number of evaluations")
    plt.ylabel("Median of best objective function values")
    plt.title(fun_label + " $\sigma = $" + str(ruido))
    filename = path_results + fun_label + "-D" + str(dim) + "-SD" + str(ruido) + "-evolution"
    plt.savefig(filename + ".pdf")
    plt.savefig(filename + ".png")
    plt.clf()

def gerar_violin_plot(final_sols, fun, fun_label, dim, ruido):
    data = [np.log(np.array(final_sols[alg]) + 1) for alg in algoritmos]
    #data = [np.array(final_sols[alg])for alg in algoritmos]

    data = np.array(data)

    df = pd.DataFrame(data.T, columns=labels)

    ax = sns.violinplot(data=df, palette=sns.color_palette(), cut=0, inner=None)
    for item in ax.collections:
        x0, y0, width, height = item.get_paths()[0].get_extents().bounds
        item.set_clip_path( plt.Rectangle((x0, y0), width/2, height, transform=ax.transData))
    num_items = len(ax.collections)
    sns.stripplot(data=df, palette=sns.color_palette(), alpha=0.4, size=7)
    for item in ax.collections[num_items:]:
        item.set_offsets(item.get_offsets() + 0.15)
    sns.boxplot(data=df, width=0.25, showfliers=False, showmeans=False, meanprops=dict(marker='o', markerfacecolor='darkorange', markersize=10, zorder=3),
    boxprops=dict(facecolor=(0,0,0,0), linewidth=3, zorder=3), whiskerprops=dict(linewidth=3), capprops=dict(linewidth=3), medianprops=dict(linewidth=3))
    plt.legend(frameon=False, fontsize=15, loc='lower left')
    # plt.title('', fontsize=28)
    plt.title(fun_label + " $\sigma = $" + str(ruido))
    plt.xlabel('Algorithms')
    plt.ylabel('Best objective function values')
    # plt.xticks(fontsize=15)
    # plt.yticks(fontsize=15)
    # sns.despine()
    filename = path_results + fun_label + "-D" + str(dim) + "-SD" + str(ruido) + "-violinplot"
    plt.savefig(filename + ".pdf")
    plt.savefig(filename + ".png")
    plt.clf()

def salvar_evolucao(amostras_mets, fun, dim, ruido):
    filename = path_results + fun.__name__ + "-" + str(dim) + "-" + str(ruido) + "-evolucao"
    with open(filename + ".pkl", "wb") as fp:
        pickle.dump(amostras_mets, fp)

ruidos = [1, 5, 10]
dims = [10]
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
metodos = [nelder_mead_base, nelder_mead_reset, rps_avg, rps_test, rps, ga, cmaes, pso]
algoritmos = ["nelder_mead_base", "nelder_mead_reset", "rps_avg", "rps_test", "rps", "ga", "cmaes", "pso"]
labels = ["NMbas", "NMrst", "RPSavg", "RPStst", "RPStau", "GA", "CMA-ES", "PSO"]

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

opts = {"lu": [(-50, 50)], "qtd": 30, "max_avals": 10000}

qtd = opts["qtd"]
max = opts["max_avals"]
lu = opts["lu"]


for fun, fun_label in zip(functions, function_labels):
    for ruido in ruidos:
        funruido = bf.adiciona_ruido(fun, desvio=ruido)
        for dim in dims:
            bounds = lu*dim
            x0s = [[(random.uniform(l, u)) for l, u in bounds] for _ in range(qtd)]
            max_avals = max*dim
            amostras_mets = {alg: [] for alg in algoritmos}
            final_sols = {alg: [] for alg in algoritmos}
            for metodo, alg in zip(metodos, algoritmos):
                parametros = params[alg][fun.__name__][dim][ruido]
                for x0 in x0s:
                    x, best_sols = metodo(funruido, x0, bounds, max_avals, **parametros, f_original = fun)
                    amostras_mets[alg].append(best_sols)
                    final_sols[alg].append(best_sols[-1])
                amostras_mets[alg] = np.median(amostras_mets[alg], axis=0)
                
            gerar_grafico_evolucao(amostras_mets, fun, fun_label, dim, ruido)
            gerar_violin_plot(final_sols, fun, fun_label, dim, ruido)
            salvar_evolucao(amostras_mets, fun, dim, ruido)
