# lib
import numpy as np
import sys
import random
from iminuit import Minuit
from iminuit.cost import LeastSquares


# Funzione di controllo degli argomenti da modificare di volta in volta nel main
def controllo_arg () :
    if len (sys.argv) != 2 :       
        print("Inserire il nome del file (compresa l'estensione) ed il valore di sigma pari a 0.3 richiesto dal tema d'esame.\n")
        sys.exit()


def funzione (x) :
    return (2 * np.sin(0.5*x + 0.78) + 0.8)


def rand_range (x_min, x_max) :
    return x_min + random.random() * (x_max - x_min)


def rand_TCL_par_gauss (mean, sigma, N) :           # par_gauss = parametri gaussiani
    y = 0.
    xMin = mean - np.sqrt(3 * N) * sigma
    xMax = mean + np.sqrt(3 * N) * sigma
    for i in range (N) :
        y += rand_range (xMin, xMax)
    y /= N 
    return y 


def funzione_fit (x, p_0, p_1, p_2, p_3) :
    return (p_0 * np.sin(p_1 * x + p_2) + p_3)


# Funzione che esegue il fit con metodo dei minimi quadrati
def esegui_fit (
        x,                  # vettore x (np.array)
        y,                  # vettore y (np.array)
        sigma,              # vettore dei sigma (np.array)
        dizionario_par,     # dizionario con parametri 
        funzione_fit        # funzione del modello da fittare
    ) :

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

