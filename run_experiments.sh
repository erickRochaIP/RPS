#! /bin/bash

if python3 --version &> /dev/null; then
    echo "Instalando pacotes necessarios"
    pip3 install -r requirements.txt --quiet

    echo "Sintonizando parametros com desvio 0.5"
    python3 sintonizar_parametros.py params-0.5 0.5 > /dev/null

    echo "Sintonizando parametros com desvio 1"
    python3 sintonizar_parametros.py params-1 1 > /dev/null

    echo "Sintonizando parametros com desvio 2"
    python3 sintonizar_parametros.py params-2 2 > /dev/null

    echo "Sintonizando parametros com desvio 5"
    python3 sintonizar_parametros.py params-5 5 > /dev/null

    echo "Sintonizando parametros com desvio 10"
    python3 sintonizar_parametros.py params-10 10 > /dev/null

    echo "Gerando experimento com desvio 0.1"
    python3 teste.py exp-0.1 0.1 > results/exp-0.1/est-ruido-0.1.txt

    echo "Gerando experimento com desvio 1"
    python3 teste.py exp-1 1 > results/exp-1/est-ruido-1.txt

    echo "Gerando experimento com desvio 2"
    python3 teste.py exp-2 2 > results/exp-2/est-ruido-2.txt

    echo "Gerando experimento com desvio 5"
    python3 teste.py exp-5 5 > results/exp-5/est-ruido-5.txt

    echo "Gerando experimento com desvio 10"
    python3 teste.py exp-10 10 > results/exp-10/est-ruido-10.txt
elif python --version &> /dev/null; then
    echo "Instalando pacotes necessarios"
    pip install -r requirements.txt --quiet

    echo "Sintonizando parametros com desvio 0.5"
    python sintonizar_parametros.py params-0.5 0.5 > /dev/null

    echo "Sintonizando parametros com desvio 1"
    python sintonizar_parametros.py params-1 1 > /dev/null

    echo "Sintonizando parametros com desvio 2"
    python sintonizar_parametros.py params-2 2 > /dev/null

    echo "Sintonizando parametros com desvio 5"
    python sintonizar_parametros.py params-5 5 > /dev/null

    echo "Sintonizando parametros com desvio 10"
    python sintonizar_parametros.py params-10 10 > /dev/null

    echo "Gerando experimento com desvio 0.1"
    python teste.py exp-0.1 0.1 > results/exp-0.1/est-ruido-0.1.txt

    echo "Gerando experimento com desvio 1"
    python teste.py exp-1 1 > results/exp-1/est-ruido-1.txt

    echo "Gerando experimento com desvio 2"
    python teste.py exp-2 2 > results/exp-2/est-ruido-2.txt

    echo "Gerando experimento com desvio 5"
    python teste.py exp-5 5 > results/exp-5/est-ruido-5.txt

    echo "Gerando experimento com desvio 10"
    python teste.py exp-10 10 > results/exp-10/est-ruido-10.txt
else
    echo "Interpretador python nao foi encontrado"
fi