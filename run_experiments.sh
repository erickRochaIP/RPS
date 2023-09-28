#! /bin/bash

echo "Sintonizando parametros"
python3 sintonizar_parametros.py > /dev/null

echo "Gerando experimento com desvio 0.1"
python3 teste.py exp-0.1 0.1 > est-ruido-0.1.txt

echo "Gerando experimento com desvio 1"
python3 teste.py exp-1 1 > est-ruido-1.txt

echo "Gerando experimento com desvio 2"
python3 teste.py exp-2 2 > est-ruido-2.txt

echo "Gerando experimento com desvio 5"
python3 teste.py exp-5 5 > est-ruido-5.txt

echo "Gerando experimento com desvio 10"
python3 teste.py exp-10 10 > est-ruido-10.txt