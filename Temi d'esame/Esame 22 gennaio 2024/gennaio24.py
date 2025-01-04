'''
python3 gennaio24.py
'''

# ----- Librerie -----

import matplotlib.pyplot as plt
import numpy as np
from lib import integral_HOM, funzione_per_normalizzare, rand_TAC, sturges, funzione_globale, media, dev_std, skewness, kurtosis, rand_TCL

# ----- Main -----

def main () :

    x_min = 0
    x_max = (3/2) * np.pi
    val_int, integral_incertezza = integral_HOM (funzione_per_normalizzare, x_min, x_max, 0, 1, 10000)

    print ("Valore integrale: ", val_int, "+/-", integral_incertezza)

    A = (val_int**(-1))
    print ("Costante di normalizzazione A: ", A)
    print ("Area normalizzata: ", A * val_int)

    N = 10000
    lista_casuali = []
    for _ in range (N) :
        lista_casuali.append (rand_TAC (funzione_globale, x_min, x_max, 1))

    Nbin = sturges (N) + 5

    bin_edges = np.linspace (x_min, x_max, Nbin)         # Regola la dimensione dei bin e Nbin = numero di bin
    fig, ax = plt.subplots (nrows = 1, ncols = 1)
    ax.hist (lista_casuali, bins=bin_edges, color = 'orange')
    ax.set_xlabel ('x')
    ax.set_ylabel ('y')
    ax.grid ()                                         
    
    print ("\nLa media è: ", media(lista_casuali))
    print ("\nLa deviazione standard è: ", dev_std(lista_casuali))

    print ("\nLa asimmetria è: ", skewness(lista_casuali))
    print ("\nLa curtosi è: ", kurtosis(lista_casuali), "\n")

    '''
    list_TCL = []
    for _ in range (N) :
        list_TCL.append (rand_TCL (x_min, x_max))
    '''
    
    plt.savefig ('Istogramma gennaio24.png')
    plt.show ()    

if __name__ == "__main__" :
    main ()