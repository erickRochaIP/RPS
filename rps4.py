import numpy as np

from simplex4 import Simplex, PontoAvaliacao


# Recebe uma funcao, ponto inicial, quantidade maxima de avaliacoes, limites de variaveis, parametros
# tolerancia, quantidade maxima de avaliacoes em um ponto
# Retorna o ponto cujo valor da funcao e o menor encontrado
def rps(f, x0, max_avals, lu, params = None, eps_x = 0.0, emax = 5):
    if params is None:
        params = {}
    coef_reflexao = params["ir"] if "ir" in params else 2
    coef_exp = params["ie"] if "ie" in params else 2
    coef_contracao = params["ic"] if "ic" in params else 1/2
    coef_encolhimento = params["is"] if "is" in params else 1/2
    
    # quantidade de vezes que melhor ponto permanece apos estagnacao
    k = 0
    
    # Set parametros inicias
    PontoAvaliacao.reset_avals()
    PontoAvaliacao.set_emax(emax)
    PontoAvaliacao.set_max_avals(max_avals)
    PontoAvaliacao.set_f(f)

    # Criar Simplex e extrair os dados
    S = Simplex(x0, lu)
    pior, segundo_pior, melhor = S.extrair_dados()
    
    # Mudar urgentemente
    melhor_dos_melhores = melhor
    atual_melhor = melhor

    while PontoAvaliacao.get_max_avals() > PontoAvaliacao.get_avals():
        # Se pior ponto esta muito proximo do melhor, reinicia simplex em volta do melhor
        if S.estagnou(eps_x):
            if atual_melhor.x == melhor.x:
                k +=1
            else:
                k = 0
            atual_melhor = melhor
            S = Simplex(atual_melhor.x, lu, k)
            pior, segundo_pior, melhor = S.extrair_dados()
            continue

        centroide = S.calcular_centroide()
        
        # Calcula ponto refletido
        refletido = PontoAvaliacao.calcular_ponto_deslocado(pior.x, centroide, coef_reflexao)
        refletido = PontoAvaliacao(refletido, lu)
        
        # Se refletido melhor que segundo pior, mas nao melhor que o melhor
        if segundo_pior > refletido and refletido >= melhor:
            S.substituir_pior_ponto(refletido)
            pior, segundo_pior, melhor = S.extrair_dados()
            continue
        # Se refletido melhor que o melhor
        if melhor > refletido:
            # Calcula expandido, e adiciona o melhor ao simplex
            expandido = PontoAvaliacao.calcular_ponto_deslocado(centroide, refletido.x, coef_exp)
            expandido = PontoAvaliacao(expandido, lu)
            if refletido > expandido:
                S.substituir_pior_ponto(expandido)
            else:
                S.substituir_pior_ponto(refletido)
            pior, segundo_pior, melhor = S.extrair_dados()
            continue
        # Se refletido for pior que o pior
        if pior >= refletido:
            contraido_externo = PontoAvaliacao.calcular_ponto_deslocado(centroide, refletido.x, coef_contracao)
            contraido_externo = PontoAvaliacao(contraido_externo, lu)
            if refletido > contraido_externo:
                S.substituir_pior_ponto(contraido_externo)
                pior, segundo_pior, melhor = S.extrair_dados()
                continue
        # Se refletido esta entre pior e segundo pior
        else:
            contraido_interno = PontoAvaliacao.calcular_ponto_deslocado(centroide, refletido.x, -coef_contracao)
            contraido_interno = PontoAvaliacao(contraido_interno, lu)
            if pior > contraido_interno:
                S.substituir_pior_ponto(contraido_interno)
                pior, segundo_pior, melhor = S.extrair_dados()
                continue

        S.contrair_simplex(coef_encolhimento, lu)
        pior, segundo_pior, melhor = S.extrair_dados()
        
    # Reset parametros
    PontoAvaliacao.reset_avals()
    PontoAvaliacao.reset_emax()
    PontoAvaliacao.reset_max_avals()
    PontoAvaliacao.reset_f()

    return melhor
