import math

import numpy as np

# Zakharov Function
def zakharov_function(x):
	D = len(x)
	somatorio05 = 0
	somatoriox2 = 0
	for i in range(D):
		somatorio05 += 0.5 * (i+1) * x[i]
		somatoriox2 += x[i] ** 2
	return somatoriox2 + somatorio05 ** 2 + somatorio05 ** 4

# Rosenbrock's Function
def rosenbrock_function(x):
	D = len(x)
	somatorio = 0
	for i in range(D-1):
		somatorio += (100 * (x[i+1] - x[i] ** 2) ** 2 + (x[i] -1) ** 2)
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
        somatorio += (x[i] ** 2 - 10 * math.cos(2*math.pi*x[i]))
    return 10 * D + somatorio

# Levy Function
def levy_function(x):
    def w(i):
        return 1 + (x[i] - 1) / (4)
    
    D = len(x)
    somatorio = 0
    for i in range(D-1):
        somatorio += ((w(i) - 1) ** 2) * (1 + 10 * math.sin(math.pi * w(i) + 1) ** 2)
    somatorio += ((w(D-1) - 1) ** 2) * (1 + math.sin(2 * math.pi * w(D-1)) ** 2)
    return math.sin(math.pi * w(0)) ** 2 + somatorio

# Bent Cigar Function
def bent_cigar_function(x):
    somatorio = 0
    for i in range(1, len(x)):
        somatorio += x[i] ** 2
    return x[0] ** 2 + 1e06 * somatorio

# HGBat Function
def hgbat_function(x):
    somatorioquadrado = 0
    somatorio = 0
    for i in range(len(x)):
        somatorioquadrado += x[i] ** 2
        somatorio += x[i]
    raiz = (abs(somatorioquadrado ** 2 - somatorio ** 2)) ** (0.5)
    sobreD = (0.5 * somatorioquadrado + somatorio) / len(x)
    return raiz + sobreD + 0.5

# High Conditioned Elliptic Function
def high_conditioned_elliptic_function(x):
    D = len(x)
    somatorio = 0
    for i in range(D):
        somatorio += (1e06) ** ((i)/(D-1)) * x[i] ** 2
    return somatorio

# Katsuura Function
def katsuura_function(x):
    D = len(x) - 1
    multiplicatorio = 1
    for i in range(len(x)):
        somatorio = 0
        for j in range(1, 33):
            somatorio += (abs(2**j*x[i] - round(2**j*x[i]))) / (2**j)
        multiplicatorio *= (1 + (i+1) * somatorio) ** (10 / D**1.2)
    return (10 / D**2) * multiplicatorio - (10 / D**2)

# Happycat Function
def happycat_function(x):
    D = len(x)
    somatorioquadrado = 0
    somatorio = 0
    for i in range(D):
        somatorioquadrado += x[i] ** 2
        somatorio += x[i]
    raiz = (abs(somatorioquadrado - D)) ** (0.25)
    sobreD = (0.5 * somatorioquadrado + somatorio) / D
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
    def z(i):
        return x[i] + 420.9687
    D = len(x)
    somatorio = 0
    for i in range(D):
        somatorio += z(i) * math.sin(math.sqrt(abs(z(i))))
    return 418.9829 * D - somatorio

# Ackley's Function
def ackleys_function(x):
    D = len(x)
    somatorioquadrado = 0
    somatoriocos = 0
    for i in range(len(x)):
        somatorioquadrado += x[i] ** 2
        somatoriocos += math.cos(2 * math.pi * x[i])
    return -20 * math.exp(-0.2*math.sqrt(somatorioquadrado/D)) - math.exp(somatoriocos / D) + 20 + math.e

# Discus Function
def discus_function(x):
    somatorioquadrado = 0
    for i in range(1, len(x)):
        somatorioquadrado += x[i] ** 2
    return 1e06 * x[0]**2 + somatorioquadrado

# Griewank Function
def griewanks_function(x):
    somatorio = 0
    multiplicatorio = 1
    for i in range(len(x)):
        somatorio += x[i]**2
        multiplicatorio *= math.cos(x[i] / math.sqrt(i+1))
    return somatorio / 4000 - multiplicatorio + 1

# Schaffer F7 Function
def schaffer_f7_function(x):
    def si(xi, xi1):
        return math.sqrt(xi ** 2 + xi1 ** 2)
    def s(i):
        return math.sqrt(x[i] ** 2 + x[i+1] ** 2)

    D=len(x)
    somatorio = 0
    for i in range(D - 1):
        si = s(i)
        somatorio += math.sqrt(si) + math.sqrt(si) * (math.sin(50 * si ** 0.2)) ** 2
    return (somatorio / (D - 1)) ** 2

def adiciona_ruido(f, media = 0, desvio = 1):
    return lambda x: f(x) + np.random.normal(media, desvio)
