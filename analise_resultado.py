import numpy as np
import matplotlib.pyplot as plt
import scikit_posthocs as sp
import scipy.stats as ss

def analisar_resultado(data_crua, path, title = ""):
    # Tirando as medias
    data = [
        [
            np.mean(data_crua[metodo][function_name])
            for function_name in range(len(data_crua[metodo]))
        ]
        for metodo in range(len(data_crua))
    ]
    data = np.array(data)
    
    # Creating plot
    fig = plt.figure(figsize =(10, 7))
    plt.boxplot(np.log(data.T + 1))
    metodos = ["rps", "nelder_mead_base", "nelder_mead_reset", "rps_avg", "rps_test"]
    plt.xticks([i for i in range(1, len(metodos)+1)], metodos)
    plt.xlabel("Algoritmos")
    plt.ylabel("Valor de objetivo")
    plt.title(title)
    plt.savefig(path + "resultado.pdf")
    plt.savefig(path + "resultado.png")
    
    # estatisticas
    print("="*20)
    print("Estatisticas (quantiles)")
    print(np.quantile(data, [0, 0.1, 0.25, 0.5, 0.75, 0.9, 1], axis=1).T)

    # posthocs
    print("="*20)
    print("Teste Friedman")
    print(ss.friedmanchisquare(*data.T))
    nf = sp.posthoc_nemenyi_friedman(data.T)
    print("="*20)
    print("Teste Nemenyi")
    print(nf)
    np.save(path + "nemenyi_friedman.npy", np.array(nf))

def analisar_arquivo(path='', file="data.npy", title=""):
    analisar_resultado(np.load(path + file),path,title)
    
if __name__ == "__main__":
    analisar_arquivo()