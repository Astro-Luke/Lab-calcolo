'''
Utilizzare la iMinuitlibreria per eseguire un adattamento sul campione simulato.

Controllare se l'adattamento è riuscito.

Stampare sullo schermo i valori dei parametri determinati e i loro sigma.
'''

import numpy as np
import matplotlib.pyplot as plt
from iminuit import Minuit
from iminuit.cost import LeastSquares

from library import rand_TCL

# ----- ----- Funzioni ----- -----

def funzione_lineare (x, m, q) :
    return m * x + q

# ----- ----- Main ----- ----- 

def main () :

    # parametri della funzione 
    m = 1.
    q = 0.

    x = np.arange(0, 10, 1)
    y = np.zeros (10)

    epsilon = np.zeros(np.size(x))
    sigma = np.zeros (np.size(x))

    for i in range (x.size) :
        epsilon[i] = rand_TCL(0, 50)
        y[i] = funzione_lineare (x[i], m, q) + epsilon[i]
    sigma = 0.5 * np.ones(np.size (y))


    least_squares = LeastSquares (x, y, sigma, funzione_lineare)
    my_minuit = Minuit (least_squares, m = 0, q = 0)    # ho messo 0 come valori iniziali
    my_minuit.migrad ()                                 # minimo dei minimi quadrati
    my_minuit.hesse ()                                  

    is_valid = my_minuit.valid
    Q_squared = my_minuit.fval
    N_dof = my_minuit.ndof

    print ("\nEsito del Fit: ", is_valid)
    print ("\nNumero di gradi di libertà: ", N_dof)
    print ("\nValore del Q-quadro: ", Q_squared, "\n")

    my_minuit.fmin
    for value, param, errore in zip (my_minuit.values, my_minuit.parameters, my_minuit.errors) : 
        print (f'{param} = {value:.6f} +/- {errore:.6f}\n')

    print("Matrice di covarianza:\n", my_minuit.covariance)

    # Calcola la retta del fit
    x_fit = np.linspace (min(x), max(x), 500)
    y_fit = funzione_lineare (x_fit, my_minuit.values[0], my_minuit.values[1])

    # Grafico con i dati e la retta del fit
    fig, ax = plt.subplots()
    ax.set_title ("Retta con errori e fit", size = 14)
    ax.set_xlabel ("x")
    ax.set_ylabel ("y")
    ax.grid ()
    ax.errorbar (x, y, xerr = 0.0, yerr = sigma, linestyle="None", marker="o", label='Dati')
    ax.plot (x_fit, y_fit, color='red', label = 'Fit lineare')
    ax.legend ()

    plt.savefig ("es11_2_3_6.png")    
    plt.show ()

if __name__ == "__main__" :
    main ()