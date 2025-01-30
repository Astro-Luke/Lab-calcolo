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