'''
python3 ottobre24.py
'''

# ----- Librerie -----

from lib import generate_gaus_bm, sturges, media, varianza_bessel, dev_std, dev_std_media, generate_gaus
import matplotlib.pyplot as plt
import numpy as np

# Lib

import random
import numpy as np
from math import ceil, sqrt

def generate_gaus_bm () :

    x1 = random.random ()
    x2 = random.random ()
    g1 = np.sqrt(-2*np.log10(x1)) * np.cos(2*np.pi*x2)
    g2 = np.sqrt(-2*np.log10(x1)) * np.sin(2*np.pi*x2)
    
    return g1, g2


def generate_gaus (mu, sigma) :

    g1, g2 = generate_gaus_bm ()
    g1 = g1*sigma + mu
    g2 = g2*sigma + mu

    return g1, g2


def sturges (N_eventi) :
    return ceil (1 + np.log2 (N_eventi))


# Media con lista
def media (lista) :
    mean = sum (lista)/len (lista)
    return mean

# Varianza con lista
def varianza_bessel (lista) :
    somma_quadrata = 0
    for elem in lista :
        somma_quadrata = somma_quadrata + (elem - media (lista))**2
    return somma_quadrata/(len (lista) - 1)

# Deviaz. standard con lista
def dev_std (lista) :
    sigma = (sqrt(varianza_bessel (lista)))
    return sigma

# Deviaz. standard della media con lista
def dev_std_media (lista) :
    return dev_std (lista)/sqrt (len (lista))


# ----- Main ----- 

def main () :

    N = 1000        # Numero di numeri pseudo-casuali da generare

    lista_casuali = []                      # inizializzo lista per numeri casuali
    for _ in range (N//2) :                    # riempimento lista
        g1, g2 = generate_gaus_bm ()
        lista_casuali.append (g1)
        lista_casuali.append (g2)

    # punto 2 esame: Istogramma
    Nbin = sturges (N)                      # per il numero di bin uso la f. di sturges
    bin_edges = np.linspace (min(lista_casuali), max(lista_casuali), Nbin)      # uso min e max
    
    fig, ax = plt.subplots (nrows = 1, ncols = 1)
    ax.hist (lista_casuali, bins = bin_edges, color = 'orange')
    ax.set_title ('Istogramma', size = 14)
    ax.set_xlabel ('x')
    ax.set_ylabel ('Conteggi')
    ax.grid ()
    
    plt.savefig ('Ottobre 2024.png')
    
    # punto 3 esame: Statistiche
    print ("Media della distribuzione: ", media (lista_casuali))
    print ("Varianza della distribuzione: ", varianza_bessel (lista_casuali))

    # punto 4 esame: mostrare che dev_standard non cambia e dev_standard della media si
    n = 1
    lista_casuali_stat = []
    sigma_list = []
    sigma_mean_list = []
    
    while (n <= 1000) :

        g1, g2 = generate_gaus_bm ()
        lista_casuali_stat.append (g1)
        lista_casuali_stat.append (g2)

        sigma_list.append (dev_std(lista_casuali_stat))
        sigma_mean_list.append (dev_std_media(lista_casuali_stat))
        n = n + 2
        
    x_axis = np.linspace (2, len(lista_casuali_stat), 500)

    # Creazione di una figura con due subplot affiancati
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))  # 1 riga, 2 colonne
    
    # Primo grafico: mostra l'andamento della deviazione standard
    axes[0].plot(x_axis, sigma_list, label="sigma") 
    axes[0].legend()
    axes[0].grid()
    axes[0].set_title("Andamento sigma")

    # Secondo grafico: mostra l'andamento della deviazione standard della media
    axes[1].plot(x_axis, sigma_mean_list, label="Sigma mean")
    axes[1].legend()
    axes[1].grid()
    axes[1].set_title("Andamento sigma media")

    plt.tight_layout()  # Questo aggiusta automaticamente spazi tra i grafici
    plt.savefig("Confronto media con sigma_media.png")

    # punto 5

    mu = 5.
    sigma = 2.
    
    list_gaus_histo = []

    for _ in range (N//2) :                    # riempimento lista
        g1, g2 = generate_gaus (mu, sigma)
        list_gaus_histo.append (g1)
        list_gaus_histo.append (g2)

    # punto 2 esame: Istogramma
    Nbin = sturges (N)                                                              # per il numero di bin uso la f. di sturges
    bin_edges = np.linspace (min(list_gaus_histo), max(list_gaus_histo), Nbin)      # uso min e max
    
    fig, ax = plt.subplots (nrows = 1, ncols = 1)
    ax.hist (list_gaus_histo, bins = bin_edges, color = 'orange')
    ax.set_title ('Istogramma 2', size = 14)
    ax.set_xlabel ('x')
    ax.set_ylabel ('Conteggi')
    ax.grid ()
    
    plt.savefig ('Ottobre 2024 gaus.png')

    plt.show()

if __name__ == "__main__" :
    main ()
