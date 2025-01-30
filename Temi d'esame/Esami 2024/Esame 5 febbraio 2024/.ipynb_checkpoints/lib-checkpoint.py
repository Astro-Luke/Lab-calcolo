import numpy as np
import random
import sys
from math import ceil
from iminuit import Minuit
from iminuit.cost import LeastSquares

def parabola (x, a, b, c) :
    return a + b*x + c*(x**2)


# Funzione sturges per il binnaggio (funziona discretamente bene, ma conviene sempre veerificare)
def sturges (N_eventi) :
    return ceil (1 + np.log2 (N_eventi))


# Distribuzione uniforme tra x_min e x_max con seed scelto in auto
def rand_range (x_min, x_max) :
    return x_min + random.random() * (x_max - x_min)


# Funzione che genera numeri pseudocasuali partendo dal teorema centrale del limite usando media, sigma di una gaussiana
# ed N numero di eventi pseudocasuali
def rand_TCL_par_gauss (mean, sigma, N) :           # par_gauss = parametri gaussiani
    y = 0. 
    xMin = mean - np.sqrt(3 * N) * sigma
    xMax = mean + np.sqrt(3 * N) * sigma
    for i in range (N) :
        y = y + rand_range (xMin, xMax)
    y /= N 
    return y 


def esegui_fit (x, y, sigma, dizionario_par, funzione_fit) :

    if not (isinstance(dizionario_par, dict)) :
        print ("Inserisci un dizionario come quarto parametro.\n")
        sys.exit()

    least_squares = LeastSquares (x, y, sigma, funzione_fit)
    my_minuit = Minuit (least_squares, **dizionario_par)
    my_minuit.migrad ()                                 
    my_minuit.hesse ()                                  

    is_valid = my_minuit.valid
    Q_squared = my_minuit.fval
    N_dof = my_minuit.ndof
    matrice_cov = my_minuit.covariance

    diz_risultati = {
        "Validità": is_valid, 
        "Qsquared": Q_squared,
        "Ndof": N_dof,
        "Param": my_minuit.parameters,
        "Value": my_minuit.values,
        "Errori": my_minuit.errors,
        "MatriceCovarianza": matrice_cov
    }

    return diz_risultati