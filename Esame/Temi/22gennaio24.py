'''
python3 gennaio24.py
'''

# ----- Librerie -----

import matplotlib.pyplot as plt
import numpy as np
from lib import integral_HOM, funzione_per_normalizzare, rand_TAC, sturges, funzione_globale, media, dev_std, skewness, kurtosis, rand_TCL

# Lib

import random
import numpy as np
from math import ceil, sqrt, pow

# Funzione sturges per il binnaggio (funziona discretamente bene, ma conviene sempre veerificare)
def sturges (N_eventi) :
    return ceil (1 + np.log2 (N_eventi))

# Distribuzione uniforme tra x_min e x_max con seed scelto in auto
def rand_range (x_min, x_max) :
    return x_min + random.random() * (x_max - x_min)

def funzione_per_normalizzare (x) :
    return (np.cos(x))**2

#Funzione per il calcolo dell'integrale (area) e scarto secondo il metodo Hit Or Miss
def integral_HOM (f, x_min, x_max, y_min, y_max, N_punti) :
    x_coord = []
    y_coord = []
    for _ in range (N_punti) :
        x_coord.append (rand_range (x_min, x_max))
        y_coord.append (rand_range (y_min, y_max))
    
    points_under = 0
    for x, y in zip (x_coord, y_coord) :             #zip per iterare su più liste in contemporanea
        if (f (x) > y) :
            points_under = points_under + 1
    
    A_rett = (x_max - x_min) * (y_max - y_min)
    frac = float (points_under) / float (N_punti)
    integral = A_rett * frac
    integral_incertezza = A_rett**2 * frac * (1-frac) / N_punti
    return integral, integral_incertezza


def funzione_globale (x) :
    A = 0.
    val_int, incert_integ = integral_HOM(funzione_per_normalizzare, 0, (3/2)*np.pi, 0, 1, 1000)
    A = A + (val_int**(-1))
    if (x > 0 and x < ((3/2) * np.pi)) :
        return A*(np.cos(x))**2
    else :
        return 0.

#Funzione che genera numeri pseudocasuali tramite l'argoritmo Try And Catch e distribuzione uniforme rand_range
def rand_TAC (f, x_min, x_max, y_max) :
    x = rand_range (x_min, x_max)
    y = rand_range (0, y_max)
    while (y > f (x)) :
        x = rand_range (x_min, x_max)
        y = rand_range (0, y_max)
    return x

# Media con lista
def media (lista) :
    mean = sum(lista)/len(lista)
    return mean

# Varianza con lista
def varianza (lista) :
    somma_quadrata = 0
    for elem in lista :
        somma_quadrata = somma_quadrata + (elem - media(lista))**2
    return somma_quadrata/(len(lista))

# Deviaz. standard con lista
def dev_std (lista) :
    sigma = sqrt(varianza(lista))
    return sigma

# Skewness con lista
def skewness(lista):
    mean = media(lista)  # Calcola la media
    sigma = dev_std(lista)  # Calcola la deviazione standard
    n = len(lista)
    somma_cubi = 0
    for elem in lista:
        somma_cubi = somma_cubi + (elem - mean)**3
    skew = somma_cubi / (n * sigma**3)
    return skew

# Curtosi con lista
def kurtosis(lista):
    mean = media(lista)  # Calcola la media
    variance = varianza(lista)  # Calcola la varianza
    n = len(lista)
    somma_quarte = 0
    for elem in lista:
        somma_quarte = somma_quarte + (elem - mean)**4
    kurt = somma_quarte / (n * variance**2) - 3
    return kurt

# Funzione che genera numeri pseudocasuali partendo dal teorema centrale del limite
def rand_TCL (xMin, xMax, N = 1000) :
    y = 0.
    for _ in range (N) :
        y = y + rand_range (xMin, xMax)
    y /= N
    return y

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
