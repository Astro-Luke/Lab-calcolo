import random
import numpy as np


def sturges (N_eventi) :
    return int (np.ceil (1 + np.log2 (N_eventi))) 

# ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ----

def rand_range (x_min, x_max) :
    return x_min + (x_max - x_min) * random.random()

# ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ----

def rand_TCL_unif (x_min, x_max, N = 1000) :
    y = 0. 
    for i in range (N) :
        y += rand_range (x_min, x_max)
    y /= N 
    return y 

# ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ----

def f (x) :
    return -((x-2)**2) + 1

# ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ----

def rand_TAC (f, x_min, x_max, y_max) :
    x = rand_range (x_min, x_max)
    y = rand_range (0, y_max)
    while (y > f (x)) :
        x = rand_range (x_min, x_max)
        y = rand_range (0, y_max)
    return x

# ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ----

def rand_TCL_para (x_min, x_max, y_max, N = 1000) :
    y = 0.
    for i in range (N) :
        y = y + rand_TAC (f, x_min, x_max, y_max)
    y /= N
    return y

# ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ----

# Media con lista
def media (lista) :
    mean = sum(lista)/len(lista)
    return mean

    
# Varianza con lista
def varianza (lista) :
    somma_quadrata = 0
    for elem in lista :
        somma_quadrata = somma_quadrata + (elem - media(lista))**2
    return somma_quadrata/(len(lista) -1)

    
# Deviaz. standard con lista
def dev_std (lista) :
    sigma = np.sqrt(varianza(lista))
    return sigma

    
# Deviaz. standard della media con lista
def dev_std_media (lista) :
    return dev_std(lista)/sqrt(len(lista))


# Skewness con lista
def skewness (lista):
    mean = media(lista)  # Calcola la media
    sigma = dev_std(lista)  # Calcola la deviazione standard
    n = len(lista)
    somma_cubi = 0
    for elem in lista:
        somma_cubi = somma_cubi + (elem - mean)**3
    skew = somma_cubi / (n * sigma**3)
    return skew


# Curtosi con lista
def kurtosis (lista):
    mean = media(lista)  # Calcola la media
    variance = varianza(lista)  # Calcola la varianza
    n = len(lista)
    somma_quarte = 0
    for elem in lista:
        somma_quarte = somma_quarte + (elem - mean)**4
    kurt = somma_quarte / (n * variance**2) - 3
    return kurt


def rand_TCL_unif_punto5 (x_min, x_max, N) :
    sample = []
    for count in range (N) :
        y = 0. 
        for i in range (N) :
            y += rand_range (x_min, x_max)
        y /= N
        sample.append (y)
    return sample 


def rand_TCL_para_punto5 (x_min, x_max, y_max, N) :
    sample = []
    for count in range (N) :
        y = 0.
        for i in range (N) :
            y = y + rand_TAC (f, x_min, x_max, y_max)
        y /= N
        sample.append (y)
    return sample