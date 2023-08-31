import numpy as np
import matplotlib.pyplot as plt
import scikit_posthocs as sp
import scipy.stats as ss

def analisar_resultado(data, path):
    # Creating plot
    fig = plt.figure(figsize =(10, 7))
    plt.boxplot(np.log(data.T + 1))
    metodos = ['RPSavg', 'RPStt', 'RPSlin', 'RPSp3', 'RPSr3', 'NMbas']
    plt.xticks([i for i in range(1, len(metodos)+1)], metodos)
    plt.savefig(path + "resultado.pdf")
    plt.savefig(path + "resultado.png")
    
    # estatisticas
    print("="*20)
    print("Estatisticas (quantiles)")
    print(np.quantile(data, [0, 0.25, 0.5, 0.75, 1], axis=1).T)

    # posthocs
    print("="*20)
    print("Teste Friedman")
    print(ss.friedmanchisquare(*data.T))
    nf = sp.posthoc_nemenyi_friedman(data.T)
    print("="*20)
    print("Teste Nemenyi")
    print(nf)
    np.save(path + "nemenyi_friedman.npy", np.array(nf))

def analisar_arquivo(path='', file="data.npy"):
    analisar_resultado(np.load(path + file))
    
if __name__ == "__main__":
    analisar_arquivo()