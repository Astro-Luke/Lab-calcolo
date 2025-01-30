import numpy as np
import random
import sys
from scipy.stats import chi2
from iminuit import Minuit
from iminuit.cost import LeastSquares


def sturges (N_eventi) :
    return int (np.ceil (1 + np.log2 (N_eventi))) 


def rand_range (xmin, xmax) :
    return xmin + (xmax - xmin) * random.random ()


# Funzione che genera numeri pseudocasuali partendo dal teorema centrale del limite usando media, sigma di una gaussiana
# ed N numero di eventi pseudocasuali
def rand_TCL_par_gauss (mean, sigma, N) :           # par_gauss = parametri gaussiani
    y = 0. ; 
    xMin = mean - np.sqrt(3 * N) * sigma
    xMax = mean + np.sqrt(3 * N) * sigma
    for i in range (N) :
        y += rand_range (xMin, xMax)
    y /= N 
    return y 


def funz (x) :
    return (x-2)**3 + 3


def funz_quadra (x) :
    return (x-2)**2 + 3


def funz_quadra_fit (x, a, b) :
    return (x-a)**2 + b


def funzione_fit (x, a, b) :
    return (x - a)**3 + b



def esegui_fit (
        x,                  # vettore x (np.array)
        y,                  # vettore y (np.array)
        sigma,              # vettore dei sigma (np.array)
        dizionario_par,     # dizionario con parametri 
        funzione_fit        # funzione del modello da fittare
    ) :

    if not (isinstance(dizionario_par, dict)) :
        print("Inserisci un dizionario come quarto parametro.\n")
        sys.exit()

    # Crea il modello LeastSquares
    least_squares = LeastSquares(x, y, sigma, funzione_fit)
    my_minuit = Minuit(least_squares, **dizionario_par)
    my_minuit.migrad()                                 
    my_minuit.hesse()                                  

    # Estrai i risultati principali
    is_valid = my_minuit.valid
    Q_squared = my_minuit.fval
    N_dof = my_minuit.ndof
    matrice_cov = my_minuit.covariance

    p_value = chi2.sf(Q_squared, N_dof)

    # Dizionario dei risultati
    diz_risultati = {
        "Validità": is_valid, 
        "Qsquared": Q_squared,
        "Ndof": N_dof,
        "Param": my_minuit.parameters,
        "Value": my_minuit.values,
        "Errori": my_minuit.errors,
        "MatriceCovarianza": matrice_cov,
        "Pvalue": p_value
    }

    return diz_risultati

