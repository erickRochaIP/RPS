from simplex import Simplex, PontoAvaliacao, MaxAvalsError

import matplotlib.pyplot as plt
from matplotlib.animation import PillowWriter
import numpy as np

def nelder_mead_base(f, x0=None, lu=None, max_avals=200,
                     dr=2, de=2, dc=0.5, ds=0.5,
                     save_gif=False, titulo=None, f_original=None):
    return rps(f, x0, lu, max_avals,
               dr, de, dc, ds, eps_x=0, emax=1, tau=1, alpha=0, teste=None,
               save_gif=save_gif, titulo=titulo, f_original=f_original)
    
def nelder_mead_reset(f, x0=None, lu=None, max_avals=200,
                     dr=2, de=2, dc=0.5, ds=0.5, eps_x=0.05,
                     save_gif=False, titulo=None, f_original=None):
    return rps(f, x0, lu, max_avals,
               dr, de, dc, ds, eps_x, emax=1, tau=1, alpha=0, teste=None,
               save_gif=save_gif, titulo=titulo, f_original=f_original)

def rps_avg(f, x0=None, lu=None, max_avals=200,
            dr=2, de=2, dc=0.5, ds=0.5, eps_x=0.0, emax=5,
            save_gif=False, titulo=None, f_original=None):
    return rps(f, x0, lu, max_avals,
               dr, de, dc, ds, eps_x, emax, tau=0, alpha=0, teste="none",
               save_gif=save_gif, titulo=titulo, f_original=f_original)

def rps_tst(f, x0=None, lu=None, max_avals=200,
            dr=2, de=2, dc=0.5, ds=0.5, eps_x=0.0, emax=5, alpha=0.05, teste=None,
            save_gif=False, titulo=None, f_original=None):
    return rps(f, x0, lu, max_avals,
               dr, de, dc, ds, eps_x, emax, tau=0, alpha=alpha, teste=teste,
               save_gif=save_gif, titulo=titulo, f_original=f_original)

# Recebe uma funcao, ponto inicial, quantidade maxima de avaliacoes, limites de variaveis, parametros
# tolerancia, quantidade maxima de avaliacoes em um ponto
# Retorna o ponto cujo valor da funcao e o menor encontrado
def rps(f, x0 = None, lu = None, max_avals = 200,
        dr = 2, de = 2, dc = 0.5, ds = 0.5, eps_x = 0.0, emax = 5, tau = 1, alpha = 0.05, teste = None,
        save_gif = False, titulo = None, f_original = None):
    coef_reflexao = dr
    coef_exp = de
    coef_contracao = dc
    coef_encolhimento = ds
    
    # quantidade de vezes que melhor ponto permanece apos estagnacao
    k = 0
    
    # Set parametros inicias
    PontoAvaliacao.reset_avals()
    PontoAvaliacao.set_emax(emax)
    PontoAvaliacao.set_max_avals(max_avals)
    PontoAvaliacao.set_f(f)
    PontoAvaliacao.set_tau(tau)
    PontoAvaliacao.set_alpha(alpha)
    PontoAvaliacao.set_teste(teste)
    
    if lu is None and x0 is None:
        lu = [(-5, 5), (-5, 5)]
        x0 = PontoAvaliacao.criar_ponto_dentro(lu)
    elif x0 is None:
        x0 = PontoAvaliacao.criar_ponto_dentro(lu)
    elif lu is None:
        lu = PontoAvaliacao.criar_lu_para(x0)
                
    xlists = []
    ylists = []

    # Criar Simplex
    S = Simplex(x0, lu)
    melhor = S.pontos[-1]
    melhor_de_todos = melhor
    
    try:
        S.ordenar_simplex()
    
        pior, segundo_pior, melhor = S.extrair_dados()
        melhor_de_todos = melhor
        atual_melhor = melhor

        while PontoAvaliacao.get_max_avals() > PontoAvaliacao.get_avals():
            if save_gif:
                xlists.append(get_list(pior, segundo_pior, melhor, 0))
                ylists.append(get_list(pior, segundo_pior, melhor, 1))
            
            if melhor_de_todos > melhor:    # Se forem o mesmo ponto vao gastar avaliacoes
                melhor_de_todos = melhor    # Talvez comparar de outra forma
            # Se pior ponto esta muito proximo do melhor, reinicia simplex em volta do melhor
            if S.estagnou(eps_x):
                if PontoAvaliacao.calcular_distancia(atual_melhor, melhor) < eps_x:
                    k += 1
                else:
                    k = 0
                atual_melhor = melhor
                S.gerar_novo_simplex(lu, k)
                S.ordenar_simplex()
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
            
    except MaxAvalsError:
        pass
    finally:
        # Reset parametros
        PontoAvaliacao.reset_avals()
        PontoAvaliacao.reset_emax()
        PontoAvaliacao.reset_max_avals()
        PontoAvaliacao.reset_f()
        PontoAvaliacao.reset_tau()
        PontoAvaliacao.reset_alpha()
        PontoAvaliacao.reset_teste()
    
    if save_gif:
        get_gif(lu, xlists, ylists, titulo, f_original)

    return melhor_de_todos


def get_gif(lu, xlists, ylists, titulo, f_original=None):
    fig, ax = plt.subplots()
    ax.set_title(titulo)
    plt.xlim(lu[0])
    plt.ylim(lu[1])
    metadata = dict(title='Simplex', artist='')
    writer = PillowWriter(fps=5, metadata=metadata)
    if f_original is not None:
        delta = 0.025
        x = np.arange(lu[0][0], lu[0][1], delta)
        y = np.arange(lu[1][0], lu[1][1], delta)
        X, Y = np.meshgrid(x, y)
        m, n = len(X), len(X[0])
        Z = [[f_original([X[i][j], Y[i][j]]) for j in range(n)] for i in range(m)]
        im = ax.imshow(Z, interpolation='bilinear', origin='lower', alpha=0.6,
               extent=(lu[0][0], lu[0][1], lu[1][0], lu[1][1]))
        CS = ax.contour(X, Y, Z)
    
    l, = plt.plot([], [], 'k-')
    with writer.saving(fig, "simplex.gif", 100):
        for (xlist, ylist) in zip(xlists, ylists):
            l.set_data(xlist, ylist)
            writer.grab_frame()
            

def get_list(p, s_p, m, i):
    return [p.x[i], s_p.x[i], m.x[i], p.x[i]]

def rps_gif(f, x0, max_avals, lu, params = None, eps_x = 0.0, emax = 5):
    if params is None:
        params = {}
    coef_reflexao = params["ir"] if "ir" in params else 2
    coef_exp = params["ie"] if "ie" in params else 2
    coef_contracao = params["ic"] if "ic" in params else 1/2
    coef_encolhimento = params["is"] if "is" in params else 1/2
    crescimento = params["crescimento"] if "crescimento" in params else 1
    
    # quantidade de vezes que melhor ponto permanece apos estagnacao
    k = 0
    
    # Set parametros inicias
    PontoAvaliacao.reset_avals()
    PontoAvaliacao.set_emax(emax)
    PontoAvaliacao.set_max_avals(max_avals)
    PontoAvaliacao.set_f(f)
    PontoAvaliacao.set_crescimento(crescimento)
    
    fig = plt.figure()
    l, = plt.plot([], [], 'k-')
    plt.xlim(lu[0])
    plt.ylim(lu[1])
    
    metadata = dict(title='Simplex', artist='')
    writer = PillowWriter(fps=5, metadata=metadata)
    

    # Criar Simplex
    S = Simplex(x0, lu)
    melhor = S.pontos[-1]
    
    try:
        S.ordenar_simplex()
    
        pior, segundo_pior, melhor = S.extrair_dados()
        atual_melhor = melhor

        with writer.saving(fig, "simplex.gif", 100):
            while PontoAvaliacao.get_max_avals() > PontoAvaliacao.get_avals():
                xlist = [pior.x[0], segundo_pior.x[0], melhor.x[0], pior.x[0]]
                ylist = [pior.x[1], segundo_pior.x[1], melhor.x[1], pior.x[1]]
                l.set_data(xlist, ylist)
                writer.grab_frame()

                # Se pior ponto esta muito proximo do melhor, reinicia simplex em volta do melhor
                if S.estagnou(eps_x):
                    if PontoAvaliacao.calcular_distancia(atual_melhor, melhor) < eps_x:
                        k +=1
                    else:
                        k = 0
                    print(melhor.x)
                    print(f(melhor.x))
                    atual_melhor = melhor
                    S.gerar_novo_simplex(lu, k)
                    S.ordenar_simplex()
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
            
    except MaxAvalsError:
        print(PontoAvaliacao.get_avals())
    finally:
        # Reset parametros
        PontoAvaliacao.reset_avals()
        PontoAvaliacao.reset_emax()
        PontoAvaliacao.reset_max_avals()
        PontoAvaliacao.reset_f()

    print(melhor.x)
    print(f(melhor.x))
    return melhor
