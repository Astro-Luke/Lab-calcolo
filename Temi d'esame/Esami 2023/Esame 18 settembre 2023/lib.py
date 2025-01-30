import numpy as np
import random

# Funzione sturges per il binnaggio (funziona discretamente bene, ma conviene sempre veerificare)
def sturges (N_eventi) :
    return int (np.ceil (1 + np.log2 (N_eventi))) 


def rand_range (x_min, x_max) :
    return x_min + (x_max - x_min) * random.random ()


def f_cauchy (M, gamma) :
    x_min = M - 3. * gamma
    x_max = M + 3. * gamma
    ymax = 1.
    x = rand_range (x_min, x_max)
    y = rand_range (0., ymax)
    while (y > (1/np.pi) * (gamma/(x-M)**2 + gamma**2)) :
        x = rand_range (x_min, x_max)
        y = rand_range (0., ymax)
    return x

def generate_cauchy (M, gamma) :
    lista_cauchy = []
    lista_count = []
    medie_cauchy = []
    std_cauchy = []
    ymax = 1.
    for i in np.arange (1, 101, 1) :
        x_min = M - i*gamma
        x_max = M + i*gamma
        x = rand_range (x_min, x_max)
        y = rand_range (0., ymax)
        while (y > (1/np.pi) * (gamma/(x-M)**2 + gamma**2)) :
            x = rand_range (x_min, x_max)
            y = rand_range (0., ymax)
        lista_cauchy.append (x)
        lista_count.append (i)
        medie_cauchy.append (np.mean(lista_cauchy))
        std_cauchy.append (np.std(lista_cauchy))
    return medie_cauchy, std_cauchy, lista_count


def rand_TCL_cauchy (gamma, M, N = 100) :
    y = 0.
    for i in range (N) :
        y = y + f_cauchy (M, gamma)
    y /= N 
    return y 