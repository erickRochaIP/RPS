import numpy as np
import matplotlib.pyplot as plt
import scikit_posthocs as sp
import scipy.stats as ss

def analisar_resultado(data):
    # Creating plot
    fig = plt.figure(figsize =(10, 7))
    plt.boxplot(data.T)
    metodos = ['RPSavg', 'RPStt', 'RPSlin', 'RPSp3', 'RPSr3', 'NMbas']
    plt.xticks([i for i in range(1, len(metodos)+1)], metodos)
    plt.savefig("resultado.pdf")
    plt.savefig("resultado.png")

    # posthocs
    print(ss.friedmanchisquare(*data.T))
    nf = sp.posthoc_nemenyi_friedman(data.T)
    print(nf)
    np.save("nemenyi_friedman.npy", np.array(nf))

def analisar_arquivo(file="data.npy"):
    analisar_resultado(np.load(file))
    
if __name__ == "__main__":
    analisar_arquivo()