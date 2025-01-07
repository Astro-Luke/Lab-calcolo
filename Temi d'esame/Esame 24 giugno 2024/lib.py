import numpy as np
import random


# Funzione sturges per il binnaggio (funziona discretamente bene, ma conviene sempre veerificare)
def sturges (N_eventi) :
    return np.ceil (1 + np.log2 (N_eventi))


# Distribuzione uniforme tra x_min e x_max con seed scelto in auto
def rand_range (x_min, x_max) :
    return x_min + random.random() * (x_max - x_min)


def rand_exp_inversa (t) :
    return -1. * np.log (1 - random.random()) * t

def rand_TAC_exp (lamb, N) :
    

def rand_TCL_par_gauss (mean, sigma, N) :           # par_gauss = parametri gaussiani
    y = 0. 
    xMin = mean - np.sqrt(3 * N) * sigma
    xMax = mean + np.sqrt(3 * N) * sigma
    for i in range (N) :
        y += rand_range (xMin, xMax)
    y /= N 
    return y 

    
def funz (x, lamb, mu, sigma, a, b) :
    return 0        # per ora zero per non dare errori