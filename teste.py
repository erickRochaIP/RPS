import random

import matplotlib.pyplot as plt
import numpy as np
import scikit_posthocs as sp
from scipy import optimize
import scipy.stats as ss

import benchmark_functions as bf
from rps import rps
from simplex import PontoAvaliacao

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
    if f_name is not None:
        print(f_name)
        print()
    elif f.__name__ is not None:
        print(f.__name__)
        print()
    
    print(x0)
    
    p_rps = rps(f, x0, max_avals,
                 lu,
                 params = {"ie": 2, "ic": 1/2, "ir": 2, "is": 1/2},
                 eps_x=1e-6).x
    p_sp = optimize.minimize(f, x0, method="Nelder-Mead",
                      bounds=lu).x
    
    return (p_rps, p_sp)
    
    print(f"RPS: x*={string_point(p_rps)}; f(x*)={media(f, p_rps):.3f}")
    print(f"Scipy NelderMead: x*={string_point(p_sp)}; f(x*)={media(f, p_sp):.3f}")
    print("==============")


functions = [
    bf.zakharov_function,
    bf.rosenbrock_function,
    bf.expanded_schaffer_function,
    bf.rastrigin_function,
    bf.levy_function
    ]

lu = [(-10, 10)]

rpsMedias = []
nmMedias = []

for function in functions:
    ruido = bf.adiciona_ruido(function)
    qtd = 3
    dim = 3
    max_avals = 400
    rpsMedia, nmMedia = 0, 0
    for i in range(qtd):
        limites_lu = lu*dim
        rpsP, nmP = test_function(ruido, criar_ponto(limites_lu), limites_lu, max_avals, function.__name__)
        rpsMedia += function(rpsP)
        nmMedia += function(nmP)
    rpsMedias.append(rpsMedia / qtd)
    nmMedias.append(nmMedia / qtd)

data = [
    rpsMedias,
    nmMedias
]

print("RPS")
print(rpsMedias)
print()
print("Nelder Mead")
print(nmMedias)

fig = plt.figure(figsize =(10, 7))
 
# Creating plot
plt.boxplot(data)
 
# show plot
plt.savefig("resultado")

data = np.array(data)

print(ss.friedmanchisquare(*data.T))
print(sp.posthoc_nemenyi_friedman(data))