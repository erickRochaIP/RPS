import math

import numpy as np
from scipy.stats import ttest_ind

class PontoAvaliacao:
    
    avals = 0

    def __init__(self, x, f, lu):
        self.x = x
        self.inviabilidade = PontoAvaliacao.obter_inviabilidade_ponto(x, lu)
        self.f_x = PontoAvaliacao.lista_f_x(f, x) if self.inviabilidade == 0 else math.inf
      
    def reset_avals():
        PontoAvaliacao.avals = 0
    
    def avaliar_funcao(f, x):
        PontoAvaliacao.avals += 1
        return f(x)
    
    def lista_f_x(f, x, n = 5):
        fs = [PontoAvaliacao.avaliar_funcao(f, x) for _ in range(n)]
        return fs

    def obter_inviabilidade_ponto(x, lu):
        return sum(PontoAvaliacao.obter_inviabilidade_var(xi, lui) for xi, lui in zip(x, lu))
    
    def obter_inviabilidade_var(xi, lui):
        li, ui = lui[0], lui[1]
        return max(0, li - xi) + max(0, xi - ui)

    # Recebe dois pontos e um coeficiente
    # Retorna um ponto deslocado A + AB*coef
    def calcular_ponto_deslocado(A, B, coef):
        return tuple(b*coef + a*(1 - coef) for a,b in zip(A, B))
    
    def calcular_distancia(A, B):
        vetor_distancia = [(mi - pi) for mi, pi in zip(A.x, B.x)]
        norma = math.sqrt(sum(d**2 for d in vetor_distancia))
        return norma
    
    def print_ponto(self):
        print("x", self.x, "f(x)", self.f_x, "inviabilidade", self.inviabilidade)
    
    def __lt__(self, other):
        if self.inviabilidade != other.inviabilidade:
            return self.inviabilidade < other.inviabilidade
        elif self.inviabilidade != 0:
            return False
        else:
            return np.mean(self.f_x) < np.mean(other.f_x)
        
    def __eq__(self, other):
        return self.inviabilidade == other.inviabilidade and np.mean(self.f_x) == np.mean(other.f_x)
        
    def __le__(self, other):
        return self == other or self < other
        
    def __gt__(self, other):
        return not (self <= other)
        
    def __ge__(self, other):
        return not (self<other)
    
    
        


class Simplex:

    def __init__(self, f, x0, lu, k = 0):
        self.pontos = Simplex.criar_simplex(f, x0, lu, k)
        self.ordenar_simplex()


    # Recebe o eixo e o ponto inicial
    # Retorna ponto inicial + base do eixo
    # se o ponto nao possui o eixo, retorna o proprio ponto
    def criar_ponto(i, x0, k):
        if i == len(x0):
            return tuple(x0)
        
        xi = list(x0[:])
        if xi[i] > 0.00025:
            xi[i] = (1 + 0.05*(1+k))*xi[i]
        else:
            xi[i] = 0.00025*(1+k)
        return xi

    # Recebe o ponto inicial
    # Retorna as bases e a origem
    def criar_pontos(x0, k):
        return [Simplex.criar_ponto(i, x0, k) for i in range(len(x0)+1)]

    # Recebe uma funcao, ponto inicial, limites das variaveis
    # Retorna um simplex ordenado, no qual os pontos sao ponto inicial + bases
    def criar_simplex(f, x0, lu, k):
        return [PontoAvaliacao(p, f, lu) for p in Simplex.criar_pontos(x0, k)]

    # Recebe um simplex
    # Ordena os pontos do simplex baseado na comparacao de PontoAvaliacao
    def ordenar_simplex(self):
        self.pontos.sort()

    # Recebe um simplex ordenado
    # Retorna o pior, segundo pior e melhor ponto
    def extrair_dados(self):
        return self.pontos[-1], self.pontos[-2], self.pontos[0]
    
    def estagnou(self, eps_x):
        pior, _, melhor = self.extrair_dados()
        norma = PontoAvaliacao.calcular_distancia(pior, melhor)
        return norma <= eps_x
    
    # Recebe um simplex ordenado
    # Retorna o ponto centroide, excluindo o pior ponto
    def calcular_centroide(self):
        n = len(self.pontos) - 1
        return tuple(sum(p.x[i] for p in self.pontos[:-1]) / n for i in range(n))

    # Recebe simplex ordenado, um ponto
    # Substitui pior ponto do simplex pelo ponto fornecido e ordena o simplex
    def substituir_pior_ponto(self, pa):
        self.pontos = self.pontos[:-1] + [pa]
        self.ordenar_simplex()

    # Recebe Simplex, coeficiente de encolhimento, funcao e limites das variaveis
    # Encolhe o Simplex e o ordena
    def contrair_simplex(self, coef, f, lu):
        n = len(self.pontos) - 1
        S_contraido = []
        S_contraido.append(self.pontos[0])
        melhor = self.pontos[0]
        for i in range(1, n+1):
            ponto = self.pontos[i]
            ponto_deslocado = PontoAvaliacao.calcular_ponto_deslocado(ponto.x, melhor.x, coef)
            S_contraido.append(PontoAvaliacao(ponto_deslocado, f, lu))
        self.pontos = S_contraido
        self.ordenar_simplex()

    def print_simplex(self):
        for pa in self.pontos:
            pa.print_ponto()
        print("================")

