#! /bin/bash

if python3 --version &> /dev/null; then
    echo "Instalando pacotes necessarios"
    pip3 install -r requirements.txt --quiet

    echo "Sintonizando parametros"
    python3 sintonizar_parametros.py experimento_oficial

    echo "Gerando experimentos"
    python3 testar_convergencia.py experimento_oficial

elif python --version &> /dev/null; then
    echo "Instalando pacotes necessarios"
    pip install -r requirements.txt --quiet

    echo "Sintonizando parametros"
    python sintonizar_parametros.py experimento_oficial

    echo "Gerando experimentos"
    python testar_convergencia.py experimento_oficial
else
    echo "Interpretador python nao foi encontrado"
fi