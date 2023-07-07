import numpy as np

from simplex1 import Simplex, PontoAvaliacao


# Recebe uma funcao, ponto inicial, quantidade de avaliacoes, limites de variaveis, parametros e tolerancia
# Retorna o ponto cujo valor da funcao e o menor encontrado
def rps(f, x0, avals_usr, lu, params = None, eps_x = 0.0):
    if params is None:
        params = {}
    coef_reflexao = params["ir"] if "ir" in params else 2
    coef_exp = params["ie"] if "ie" in params else 2
    coef_contracao = params["ic"] if "ic" in params else 1/2
    coef_encolhimento = params["is"] if "is" in params else 1/2
    
    # quantidade de vezes que melhor ponto permanece apos estagnacao
    k = 0
    
    PontoAvaliacao.reset_avals()

    S = Simplex(f, x0, lu)
    pior, segundo_pior, melhor = S.extrair_dados()
    
    melhor_dos_melhores = melhor
    atual_melhor = melhor

    while avals_usr > PontoAvaliacao.avals:
        if S.estagnou(eps_x):
            if atual_melhor == melhor:
                k +=1
            else:
                k = 0
            melhor_dos_melhores = min(melhor_dos_melhores, atual_melhor, melhor)
            atual_melhor = melhor
            S = Simplex(f, atual_melhor.x, lu, k)
            pior, segundo_pior, melhor = S.extrair_dados()
            continue

        centroide = S.calcular_centroide()
        refletido = PontoAvaliacao.calcular_ponto_deslocado(pior.x, centroide, coef_reflexao)
        refletido = PontoAvaliacao(refletido, f, lu)
        if segundo_pior > refletido >= melhor:
            S.substituir_pior_ponto(refletido)
            pior, segundo_pior, melhor = S.extrair_dados()
            continue
        if melhor > refletido:
            expandido = PontoAvaliacao.calcular_ponto_deslocado(centroide, refletido.x, coef_exp)
            expandido = PontoAvaliacao(expandido, f, lu)
            if refletido > expandido:
                S.substituir_pior_ponto(expandido)
            else:
                S.substituir_pior_ponto(refletido)
            pior, segundo_pior, melhor = S.extrair_dados()
            continue
        if pior >= refletido:
            contraido_externo = PontoAvaliacao.calcular_ponto_deslocado(centroide, refletido.x, coef_contracao)
            contraido_externo = PontoAvaliacao(contraido_externo, f, lu)
            if refletido > contraido_externo:
                S.substituir_pior_ponto(contraido_externo)
                pior, segundo_pior, melhor = S.extrair_dados()
                continue
        else:
            contraido_interno = PontoAvaliacao.calcular_ponto_deslocado(centroide, refletido.x, -coef_contracao)
            contraido_interno = PontoAvaliacao(contraido_interno, f, lu)
            if pior > contraido_interno:
                S.substituir_pior_ponto(contraido_interno)
                pior, segundo_pior, melhor = S.extrair_dados()
                continue

        S.contrair_simplex(coef_encolhimento, f, lu)
        pior, segundo_pior, melhor = S.extrair_dados()

    return min(melhor_dos_melhores, atual_melhor, melhor)