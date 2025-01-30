# Libreria esame 22 settembre 2022

import numpy as np
from iminuit import Minuit
from iminuit.cost import LeastSquares

def parabola (x, a, b, c) :
    return a * (x**2) + b * x + c


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


#Funzione aurea per ricerca del massimo
# qui ho fatto prendere anche un dizionario dato che ho più parametri (utile da mettere in library o portare all'esame come esempio)
def sezione_aurea_max (
    f,                      # funzione di cui trovare lo zero
    x0,                     # estremo dell'intervallo
    x1,                     # altro estremo dell'intervallo
    diz_para,
    precision = 0.0001) :   # precisione della funzione

    r = 0.618
    x2 = 0.
    x3 = 0.
    larghezza = abs (x1 - x0)
     
    while (larghezza > precision):
        x2 = x0 + r * (x1 - x0)
        x3 = x0 + (1. - r) * (x1 - x0)
      
        # si restringe l'intervallo tenendo fisso uno dei due estremi e spostando l'altro
        if (f (x3, *diz_para) < f (x2, *diz_para)) :
            x0 = x3
        else :
            x1 = x2
        larghezza = abs (x1-x0)
    return (x0 + x1) / 2.


# Main esame 22 settembre 2022
'''
python3 26settembre22.py
'''

import numpy as np
import matplotlib.pyplot as plt

from lib import parabola, esegui_fit, sezione_aurea_max

def main () :

    x_coord, y_coord = np.loadtxt ("coordinate.txt", unpack = True)
    sigma = np.ones (len(x_coord))
    # dizionario per usare la funzione di fit (esegui_fit)

    diz_para = {
        "a": 1.,
        "b": 1., 
        "c": 1.
    }

    diz_result = esegui_fit (x_coord, y_coord, sigma, diz_para, parabola)
    
    # Stampa dei valori
    print ("\nEsito del Fit: ", diz_result["Validità"])
    print ("\nValore del Q-quadro: ", diz_result["Qsquared"])
    print ("\nNumero di gradi di libertà: ", diz_result["Ndof"], "\n")
    print("Matrice di covarianza:\n", diz_result["MatriceCovarianza"])

    for param, value, errore in zip (diz_result["Param"], diz_result["Value"], diz_result["Errori"]) : 
        print (f'{param} = {value:.6f} +/- {errore:.6f}\n')

    x_fit = np.linspace (min (x_coord), max (x_coord), 500)
    y_fit = parabola (x_fit, *diz_result["Value"])

    # Punto 5
    x_max = sezione_aurea_max (parabola, min(x_coord), max(x_coord), diz_result["Value"])
    #print (x_max)
    y_max = parabola (x_max, *diz_result["Value"])

    # dal libro di fisica 1: y(x) = tg(theta) * x - (g / (2 v**2 * cos(theta)**2)) x**2

    a = diz_result["Value"][0]
    b = diz_result["Value"][1]

    gittata = -(1/a)*b
    print ("La gittata è: ", gittata)
    
    fig, ax = plt.subplots ()
    ax.scatter (x_coord, y_coord, marker = 'o', label = 'coordinate')
    ax.plot (x_fit, y_fit, color = 'red', label = 'fit parabolico')
    ax.scatter (x_max, y_max, color = 'black', marker = 'x', label = 'punto di massimo')
    ax.set_xlabel ("x")
    ax.set_ylabel ("y")
    ax.legend ()
    ax.grid ()
    plt.savefig ("traiettoria.png")
    plt.show ()
    
if __name__ == '__main__' :
    main ()