import os
import pickle
import sys

def load_file(filename):
    try:
        with open(filename, "rb") as fp:
            content = pickle.load(fp)
    except:
        print("Parametros de " + filename + " nao encontrados")
    return content

def print_nemenyi_table(data):
    labels = ["NMrst", "RPS", "GA", "CMA-ES", "PSO"]
    print("\\begin{table}[h!]")
    print("\\begin{tabular}{lccccc}")
    print("\\toprule")
    print(" & \\textbf{NMrst} & \\textbf{RPS} & \\textbf{GA} & \\textbf{CMA-ES} \\\\")
    print("\\midrule")

    for i in range(1, len(data)):
        print("\\textbf{" + labels[i] + '}', end=' ')
        for j in range(len(data)-1):
            print('&', end=' ')
            if j >= i:
                print('---', end=' ')
            else:
                if data[i][j] <= 0.05:
                    print('\\textbf{', end='')
                    print("{:7.6f}".format(data[i][j]), end='')
                    print('}', end=' ')
                else:
                    print("{:7.6f}".format(data[i][j]), end=' ')
            
        print('\\\\')




    print("\\bottomrule")
    print("\\end{tabular}")
    print("\\end{table}")



data = load_file('final-nemenyi_friedman.pkl')
print_nemenyi_table(data)
