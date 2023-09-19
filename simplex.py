import math

import numpy as np
from scipy.stats import ttest_ind

class PontoAvaliacao:
    
    _avals = 0
    _emax = 5
    _max_avals = 400
    _crescimento = 1
    _f = None
      
    # Acessar avals
    def get_avals():
        return PontoAvaliacao._avals
    
    def reset_avals():
        PontoAvaliacao._avals = 0
        
    # Acessar emax
    def get_emax():
        return PontoAvaliacao._emax
    
    def set_emax(emax):
        PontoAvaliacao._emax = max(1, emax)
    
    def reset_emax():
        PontoAvaliacao._emax = 5
        
    # Acessar max_avals
    def get_max_avals():
        return PontoAvaliacao._max_avals
    
    def set_max_avals(max_avals):
        PontoAvaliacao._max_avals = max(1, max_avals)
        
    def reset_max_avals():
        PontoAvaliacao._max_avals = 400
    
    # Acessar crescimento
    def get_crescimento():
        return PontoAvaliacao._crescimento
    
    def set_crescimento(crescimento):
        PontoAvaliacao._crescimento = max(0, crescimento)
    
    def reset_crescimento():
        PontoAvaliacao._crescimento = 1
        
    # Acessar f
    def get_f():
        return PontoAvaliacao._f
    
    def set_f(f):
        PontoAvaliacao._f = f
        
    def reset_f():
        PontoAvaliacao._f = None
        
    def __init__(self, x, lu):
        self.x = x
        self.inviabilidade = PontoAvaliacao.obter_inviabilidade_ponto(x, lu)
        self.f_x = [] if self.inviabilidade == 0 else math.inf
    
    # Avaliar funcao
    def avaliar_funcao(x):
        if PontoAvaliacao._avals >= PontoAvaliacao._max_avals:
            raise MaxAvalsError(PontoAvaliacao._avals)
        PontoAvaliacao._avals += 1
        return PontoAvaliacao._f(x)

    # Obter inviabilidade
    def obter_inviabilidade_ponto(x, lu):
        return sum(PontoAvaliacao.obter_inviabilidade_var(xi, lui) for xi, lui in zip(x, lu))
    
    def obter_inviabilidade_var(xi, lui):
        li, ui = lui[0], lui[1]
        return max(0, li - xi) + max(0, xi - ui)

    # Recebe dois pontos e um coeficiente
    # Retorna um ponto deslocado A + AB*coef
    def calcular_ponto_deslocado(A, B, coef):
        return tuple(b*coef + a*(1 - coef) for a,b in zip(A, B))
    
    # Recebe dois pontos
    # Retorna a norma do vetor que une os pontos
    def calcular_distancia(A, B):
        vetor_distancia = [(mi - pi) for mi, pi in zip(A.x, B.x)]
        norma = math.sqrt(sum(d**2 for d in vetor_distancia))
        return norma
    
    def print_ponto(self):
        print("x", self.x, "f(x)", self.f_x, "inviabilidade", self.inviabilidade)
    
    # Comparacoes entre pontos
    def __lt__(self, other):
        # Se os dois tem inviabilidades diferentes
        if self.inviabilidade != other.inviabilidade:
            return self.inviabilidade < other.inviabilidade
        # Se os dois sao igualmente inviaveis
        elif self.inviabilidade != 0:
            return False
        # Se os dois sao viaveis
        else:
            R = PontoAvaliacao.confidence_compare(self, other)
            return R == 1
        
    def __eq__(self, other):
        # Se os dois tem inviabilidades diferentes
        if self.inviabilidade != other.inviabilidade:
            return False
        # Se os dois sao igualmente inviaveis
        elif self.inviabilidade != 0:
            return True
        # Se os dois sao viaveis
        else:
            R = PontoAvaliacao.confidence_compare(self, other)
            return R == 0
        
    def __le__(self, other):
        # Se os dois tem inviabilidades diferentes
        if self.inviabilidade != other.inviabilidade:
            return self.inviabilidade < other.inviabilidade
        # Se os dois sao igualmente inviaveis
        elif self.inviabilidade != 0:
            return True
        # Se os dois sao viaveis
        else:
            R = PontoAvaliacao.confidence_compare(self, other)
            return R == 1 or R == 0
        
    def __gt__(self, other):
        # Se os dois tem inviabilidades diferentes
        if self.inviabilidade != other.inviabilidade:
            return self.inviabilidade > other.inviabilidade
        # Se os dois sao igualmente inviaveis
        elif self.inviabilidade != 0:
            return False
        # Se os dois sao viaveis
        else:
            R = PontoAvaliacao.confidence_compare(self, other)
            return R == -1
        
    def __ge__(self, other):
        # Se os dois tem inviabilidades diferentes
        if self.inviabilidade != other.inviabilidade:
            return self.inviabilidade > other.inviabilidade
        # Se os dois sao igualmente inviaveis
        elif self.inviabilidade != 0:
            return True
        # Se os dois sao viaveis
        else:
            R = PontoAvaliacao.confidence_compare(self, other)
            return R == -1
    
    # Realiza Teste de T Student entre duas amostras a b, com confianca alpha
    # Se a < b retorna 1
    # Se a = b retorna 0
    # Se a > b retorna -1
    def tTest(a, b, alpha = 0.05):
        if ttest_ind(a, b, alternative="less").pvalue < alpha:
            return 1
        if ttest_ind(a, b, alternative="greater").pvalue < alpha:
            return -1
        return 0
    
    # Compara dois pontos a e b, com viabilidade = 0
    # Se a < b retorna 1
    # Se a = b retorna 0
    # Se a > b retorna -1
    def confidence_compare(a, b):
        e = math.ceil(PontoAvaliacao._emax * (PontoAvaliacao._avals / PontoAvaliacao._max_avals)**PontoAvaliacao._crescimento)
        
        # Se e for 1, compara um unico valor de f(x) para a e b
        if e <= 1:
            # Se algum dos pontos nao foi avaliado ainda, avalie
            if a.f_x == []:
                a.f_x.append(PontoAvaliacao.avaliar_funcao(a.x))
            if b.f_x == []:
                b.f_x.append(PontoAvaliacao.avaliar_funcao(b.x))
                
            ma, mb = a.f_x[0], b.f_x[0]
            if ma < mb:
                return 1
            if ma == mb:
                return 0
            if ma > mb:
                return -1
            
        # Se e > 1, avalia funcao nos pontos ate que ambos
        # tenham pelo menos duas avaliacoes
        while(len(a.f_x) < 2):
            a.f_x.append(PontoAvaliacao.avaliar_funcao(a.x))
        while(len(b.f_x) < 2):
            b.f_x.append(PontoAvaliacao.avaliar_funcao(b.x))
        
        # Extrair tamanho, media, desvio padrao dos pontos
        na, ma, sa = len(a.f_x), np.mean(a.f_x), np.std(a.f_x)
        nb, mb, sb = len(b.f_x), np.mean(b.f_x), np.std(b.f_x)
        
        # Realizar um teste t
        H = PontoAvaliacao.tTest(a.f_x, b.f_x)
        
        # Enquanto o teste alegar igualdade e for possivel avaliar algum ponto
        while(H == 0 and (na < e or nb < e)):
            # Se for possivel avaliar a e o desvio de a for maior que b
            #   ou se nao for possivel avaliar b
            # Entao avalia a
            if (na < e and sa > sb) or (nb == e):
                a.f_x.append(PontoAvaliacao.avaliar_funcao(a.x))
            # Caso contrario, sabemos que nb < e e desvio de b maior que a
            # Entao avalia b
            else:
                b.f_x.append(PontoAvaliacao.avaliar_funcao(b.x))
                
            # Extrair tamanho, media, desvio padrao dos pontos
            na, ma, sa = len(a.f_x), np.mean(a.f_x), np.std(a.f_x)
            nb, mb, sb = len(b.f_x), np.mean(b.f_x), np.std(b.f_x)
            
            # Realizar um teste t
            H = PontoAvaliacao.tTest(a.f_x, b.f_x)
        
        # Se o teste acabou em igualdade, compara as medias
        if H == 0:
            if ma < mb:
                return 1
            if ma == mb:
                return 0
            if ma > mb:
                return -1
        # Senao retorna o resultado do teste
        else:
            return H
        

class Simplex:

    def __init__(self, x0, lu, k = 0):
        self.pontos = Simplex.criar_simplex(x0, lu, k)

    # Recebe ponto inicial, limites das variaveis, repeticao do melhor
    # Retorna um simplex, no qual os pontos sao ponto inicial + bases
    def criar_simplex(x0, lu, k):
        return [PontoAvaliacao(p, lu) for p in Simplex.criar_pontos(x0, k)]
    
    # Recebe o ponto inicial, repeticao do melhor
    # Retorna coordenadas do ponto inicial + bases
    def criar_pontos(x0, k):
        return [Simplex.criar_ponto(i, x0, k) for i in range(len(x0)+1)]
    
    # Recebe i, ponto inicial, repeticao do melhor
    # Retorna ponto i-esimo do simplex
    def criar_ponto(i, x0, k):
        if i == len(x0):
            return tuple(x0)
        
        xi = list(x0[:])
        xi[i] = xi[i] +1+k
        
        return xi
    
    def gerar_novo_simplex(self, lu, k):
        self.pontos = [self.pontos[0]] + [PontoAvaliacao(p, lu) for p in Simplex.criar_pontos_redor(self.pontos[0].x, k)]

    def criar_pontos_redor(x0, k):
        return [Simplex.criar_ponto(i, x0, k) for i in range(len(x0))]

    # Recebe um simplex
    # Ordena os pontos do simplex baseado na comparacao de PontoAvaliacao
    def ordenar_simplex(self):
        self.pontos.sort()

    # Recebe um simplex ordenado
    # Retorna o pior, segundo pior e melhor ponto
    def extrair_dados(self):
        return self.pontos[-1], self.pontos[-2], self.pontos[0]
    
    def estagnou(self, eps_x): #modificar para quando o ponto estagnar, gerar um novo ponto aleatório
        pior, _, melhor = self.extrair_dados()
        norma = PontoAvaliacao.calcular_distancia(pior, melhor)
        return norma <= eps_x
    
    # Recebe um simplex ordenado
    # Retorna o ponto centroide, excluindo o pior ponto
    def calcular_centroide(self):
        n = len(self.pontos) - 1
        return tuple(sum(p.x[i] for p in self.pontos[:-1]) / n for i in range(n))

    # Recebe simplex ordenado, um ponto
    # Substitui pior ponto do simplex pelo ponto fornecido
    def substituir_pior_ponto(self, pa):
        i = 0
        self.pontos.pop()
        try:
            for p in self.pontos:
                if p > pa:
                    break
                i += 1
        finally:
            self.pontos.insert(i, pa)

    # Recebe simplex ordenado, coeficiente de encolhimento, funcao e limites das variaveis
    # Encolhe o Simplex e o ordena
    def contrair_simplex(self, coef, lu):
        n = len(self.pontos) - 1
        S_contraido = []
        S_contraido.append(self.pontos[0])
        melhor = self.pontos[0]
        for i in range(1, n+1):
            ponto = self.pontos[i]
            ponto_deslocado = PontoAvaliacao.calcular_ponto_deslocado(ponto.x, melhor.x, coef)
            S_contraido.append(PontoAvaliacao(ponto_deslocado, lu))
        self.pontos = S_contraido
        self.ordenar_simplex()

    def print_simplex(self):
        for pa in self.pontos:
            pa.print_ponto()
        print("================")


class MaxAvalsError(Exception):
    def __init__(self, avals=None):
        self.message = "Quantidade maxima de avaliacoes atingida."
        if avals is not None:
            self.message += " (" + str(avals) + " avaliacoes)"
        super().__init__(self.message)
