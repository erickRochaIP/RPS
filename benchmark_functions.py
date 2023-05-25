import math

import numpy as np

# Zakharov Function
def zakharov_function(x):
	D = len(x)
	somatorio05 = 0
	somatoriox2 = 0
	for i in range(D):
		somatorio05 += 0.5 * x[i]
		somatoriox2 += x[i] ** 2
	return somatoriox2 + somatorio05 ** 2 + somatorio05 ** 4

# Rosenbrock's Function
def rosenbrock_function(x):
	D = len(x) - 1
	somatorio = 0
	for i in range(D):
		somatorio += (100 * (x[i] ** 2 - x[i+1]) ** 2 + (x[i+1] -1) ** 2)
	return somatorio

# Expanded Schaffer's Functon
def expanded_schaffer_function(x):
	def g(x, y):
		return 0.5 + (math.sin(math.sqrt(x**2 + y**2)) ** 2 - 0.5) / ((1 + 0.001 * (x**2 + y**2)) ** 2)

	D = len(x)
	somatorio = 0
	for i in range(D-1):
		somatorio += g(x[i], x[i+1])
	somatorio += g(x[D-1], x[0])
	return somatorio
		

# Rastrigin's Function
def rastrigin_function(x):
    D = len(x)
    somatorio = 0
    for i in range(D):
        somatorio += (x[i] ** 2 - 10 * math.cos(2*math.pi*x[i]) + 10)
    return somatorio

# Levy Function
def levy_function(x):
    def wi(xi):
        return 1 + (xi - 1) / (4)
    
    D = len(x) - 1
    somatorio = 0
    for i in range(D):
        somatorio += ((wi(x[i]) - 1) ** 2) * (1 + 10 * math.sin(math.pi * wi(x[i]) - 1) ** 2)
        somatorio += ((wi(x[D]) - 1) ** 2) * (1 + math.sin(2 * math.pi * wi(x[D])) ** 2)
    somatorio += math.sin(math.pi * wi(x[0])) ** 2
    return somatorio

def adiciona_ruido(f, media = 0, desvio = 1):
    return lambda x: f(x) + np.random.normal(media, desvio)
