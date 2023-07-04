import numpy as np
from scipy import optimize

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

def test_function(f, x0, lu, f_name=None):
    if f_name is not None:
        print(f_name)
        print()
    elif f.__name__ is not None:
        print(f.__name__)
        print()
        
    
    p_rps = rps(f, x0, 400,
                 lu,
                 params = {"ie": 2, "ic": 1/2, "ir": 2, "is": 1/2},
                 eps_x=1e-6).x
    p_sp = optimize.minimize(f, x0, method="Nelder-Mead",
                      bounds=lu).x
    
    print(f"RPS: x*={string_point(p_rps)}; f(x*)={media(f, p_rps):.3f}")
    print(f"Scipy NelderMead: x*={string_point(p_sp)}; f(x*)={media(f, p_sp):.3f}")
    print("==============")
    
    # TODO: retornar tupla com as melhores solucoes


functions = [
    bf.zakharov_function,
    bf.rosenbrock_function,
    bf.expanded_schaffer_function,
    bf.rastrigin_function,
    bf.levy_function
    ]

x0 = [-3]
lu = [(-10, 10)]

# TODO: criar matriz fxk (np.array)
for function in functions:
    ruido = bf.adiciona_ruido(function)
    # TODO: chamar essa funcao 10 vezes
    test_function(ruido, x0*3, lu*3, function.__name__)
    # TODO: para cada metodo, retornar a media das 10 solucoes

# TODO: gerar boxplot
# TODO: aplicar testes Friedman e Nemenyi