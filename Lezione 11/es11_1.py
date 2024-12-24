'''
Dopo aver definito, in una libreria dedicata, una funzione lineare
con due parametri

Scrivi un programma che genera un set di 10 coppie in modo che i punti
sono distribuiti casualmente lungo l'asse orizzontale tra 0 e 10, e i punti
sono costruiti utilizzando la formula

Rappresentare graficamente il campione ottenuto, incluse le barre di errore previste.
'''

import numpy as np
import matplotlib.pyplot as plt

from library import rand_TCL

# ----- ----- Funzioni ----- -----

def funzione_lineare (x, m, q) :
    return m * x + q

# ----- ----- Main ----- ----- 

def main () :

    # parametri della funzione 
    m = 1.
    q = 0.

    x = np.arange(0, 10, 1)
    y = np.zeros (10)

    epsilon = np.zeros(np.size(x))
    sigma = np.zeros (np.size(x))

    for i in range (x.size) :
        epsilon[i] = rand_TCL(0, 50)
        y[i] = funzione_lineare (x[i], m, q) + epsilon[i]
    sigma = 0.5 * np.ones(np.size (y))

    # grafico
    fig, ax = plt.subplots ()
    ax.set_title ('retta con errori', size = 14)
    ax.set_xlabel ('x')
    ax.set_ylabel ('y')
    ax.errorbar (x, y, xerr = 0.0, yerr = sigma, linestyle = 'None', marker = 'o') 
    plt.savefig ("es11_1.png")
    plt.show ()

if __name__ == "__main__" :
    main ()