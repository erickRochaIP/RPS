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

# Bent Cigar Function
def bent_cigar_function(x):
    somatorio = 0
    for i in range(1, len(x)):
        somatorio += x[i] ** 2
    somatorio *= 1e06
    somatorio += x[0] ** 2
    return somatorio

# HGBat Function
def hgbat_function(x):
    somatorioquadrado = 0
    somatorio = 0
    for i in range(len(x)):
        somatorioquadrado += x[i] ** 2
        somatorio += x[i]
    raiz = (math.abs(somatorioquadrado ** 2 - somatorio ** 2)) ** (0.5)
    sobreD = (0.5 * somatorioquadrado + somatorio) / len(x)
    return raiz + sobreD + 0.5

# High Conditioned Elliptic Function
def high_conditioned_elliptic_function(x):
    D = len(x) - 1
    somatorio = 0
    for i in range(len(x)):
        somatorio += (1e06) ** ((i)/(D-1)) * x[i] ** 2
    return somatorio

# Katsuura Function
def katsuura_function(x):
    D = len(x) - 1
    multiplicatorio = 1
    for i in range(len(x)):
        somatorio = 0
        for j in range(1, 33):
            somatorio += (math.abs(2**j*x[i] - round(2**j*x[i]))) / (2**j)
        multiplicatorio *= (1 + (i+1) * somatorio) ** (10 / D**1.2)
    return (10 / D**2) * multiplicatorio - (10 / D**2)

# Happycat Function
def happycat_function(x):
    somatorioquadrado = 0
    somatorio = 0
    for i in range(len(x)):
        somatorioquadrado += x[i] ** 2
        somatorio += x[i]
    raiz = (math.abs(somatorioquadrado - len(x))) ** (0.25)
    sobreD = (0.5 * somatorioquadrado + somatorio) / len(x)
    return raiz + sobreD + 0.5

# Expanded Rosenbrocks plus Griewangk Function
def expanded_rosenbrocks_plus_griewangk_function(x):
    somatorio = 0
    for i in range(len(x) - 1):
        somatorio += griewanks_function([rosenbrock_function([x[i], x[i+1]])])
    somatorio += griewanks_function([rosenbrock_function([x[len(x)-1], x[0]])])
    return somatorio

# Modified Schwefels Function
def modified_schwefels_function(x):
    def g(z):
        if z > 500:
            return (500 - z % 500) * math.sin(math.sqrt(math.abs((500 - z % 500)))) - (z - 500) ** 2 / (1000*len(x))
        elif z < -500:
            absz = -z
            return (absz % 500 - 500) * math.sin(math.sqrt(math.abs((absz % 500 - 500)))) - (z + 500) ** 2 / (1000*len(x))
        else:
            return z * math.sin(math.sqrt(math.abs(z)))
    D = len(x) - 1
    somatorio = 0
    for i in range(len(x)):
        somatorio += g(x[i] + 4.209687462275036e002)
    return 418.9829 * D - somatorio

# Ackley's Function
def ackleys_function(x):
    somatorioquadrado = 0
    somatoriocos = 0
    for i in range(len(x)):
        somatorioquadrado += x[i] ** 2
        somatoriocos += math.cos(2 * math.pi * x[i])
    return -20 * math.exp(-0.2*math.sqrt(somatorioquadrado/len(x))) - math.exp(somatoriocos / len(x)) + 20 + math.e

# Discus Function
def discus_function(x):
    somatorioquadrado = 0
    for i in range(len(x)):
        somatorioquadrado += x[i] ** 2
    return 1e06 * x[0]**2 + somatorioquadrado

# Griewank Function
def griewanks_function(x):
    somatorio = 0
    multiplicatorio = 1
    for i in range(len(x)):
        somatorio += x[i]**2 / 400
        multiplicatorio *= math.cos(x[i] / math.sqrt(i+1))
    return somatorio - multiplicatorio + 1

# Schaffer F7 Function
def schaffer_f7_function(x):
    def si(xi, xi1):
        return math.sqrt(xi ** 2 + xi1 ** 2)
    
    somatorio = 0
    for i in range(len(x) - 1):
        somatorio += math.sqrt(si(x[i], x[i+1])) * (math.sin(50 * si(x[i], x[i+1]) ** 0.2) + 1)
    return (somatorio / (len(x) - 1)) ** 2

def adiciona_ruido(f, media = 0, desvio = 1):
    return lambda x: f(x) + np.random.normal(media, desvio)
