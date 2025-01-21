'''
python3 febbraio24.py
'''

import numpy as np
import matplotlib.pyplot as plt
from iminuit import Minuit
from iminuit.cost import LeastSquares
from lib import parabola, rand_TCL_par_gauss, rand_range, esegui_fit, sturges


def main () :

    # parametri
    a, b, c = 3, 2, 1
    diz_par = {
        "a": 3,
        "b": 2,
        "c": 1
    }

    # Intervallo e numero di punti
    x_min = 0
    x_max = 10
    N_punti = 10
    
    # creazione array 
    x = np.zeros (N_punti)
    epsilon = np.zeros (N_punti)
    y = np.zeros (N_punti)

    for i in range (N_punti) :
        x[i] = rand_range(x_min, x_max)
        epsilon[i] = rand_TCL_par_gauss (0, 10, 10)
        y[i] = parabola (x[i], a, b, c) + epsilon[i]

    sigma = np.full (1, 10)

    diz_result = esegui_fit (x, y, sigma, diz_par, parabola)

    print ("\nEsito del Fit: ", diz_result["Validità"])
    print ("\nNumero di gradi di libertà: ", diz_result["Qsquared"])
    print ("\nValore del Q-quadro: ", diz_result["Ndof"], "\n")

    print("Matrice di covarianza:\n", diz_result["MatriceCovarianza"])

    for param, value, errore in zip (diz_result["Param"], diz_result["Value"], diz_result["Errori"]) : 
        print (f'{param} = {value:.6f} +/- {errore:.6f}\n')

    # Calcola la parabola del fit
    x_fit = np.linspace (min(x), max(x), 500)
    y_fit = parabola (x_fit, *diz_result["Value"])

    # grafico
    fig, ax = plt.subplots ()
    ax.set_title ('Parabola con errori e fit', size = 14)
    ax.set_xlabel ('x')
    ax.set_ylabel ('y')
    ax.errorbar (x, y, xerr = 0.0, yerr = 10, linestyle = 'None', marker = 'o') 
    ax.plot (x_fit, y_fit, color = 'red', label = 'Fit')
    ax.grid ()

    plt.savefig ("esame febbraio24 fit.png")

    N_toy = 1000
    lista_Q2 = []
    for _ in range (N_toy) :
        for i in range (N_punti) :
            x[i] = rand_range(x_min, x_max)
            epsilon[i] = rand_TCL_par_gauss (0, 10, 10)
            y[i] = parabola (x[i], a, b, c) + epsilon[i]

        diz_result_Q2 = esegui_fit (x, y, sigma, diz_par, parabola)
        lista_Q2.append (diz_result_Q2["Qsquared"])
    

    # punto 5
    lista_Q2_epsilon_uniform = []    
    for _ in range (N_toy) :
        for i in range (N_punti) :
            x[i] = rand_range (x_min, x_max)
            epsilon[i] = rand_range (-10*np.sqrt(3), 10*np.sqrt(3))
            y[i] = parabola (x[i], a, b, c) + epsilon[i]

        diz_Q2_epsilon_unif = esegui_fit (x, y, sigma, diz_par, parabola)
        lista_Q2_epsilon_uniform.append (diz_Q2_epsilon_unif["Qsquared"])
    
    Nbin = sturges (N_toy)

    bin_edges = np.linspace(min(lista_Q2), max(lista_Q2), Nbin)         # Regola la dimensione dei bin e Nbin = numero di bin
    fig, ax = plt.subplots (nrows = 1, ncols = 1)
    ax.hist (lista_Q2, bins=bin_edges, color = 'orange', label='epsilon TCL gaus')  # Spesso conviene usare bins = 'auto' evitando di scrivere la linea di codice con bin_edges, per farlo però bisogna importare numpy
    ax.hist (lista_Q2_epsilon_uniform, bins = bin_edges, color = 'blue', histtype = 'step', label='epsilon uniformi')
    ax.set_title ('Istogramma Q2', size = 14)
    ax.set_xlabel ('x')
    ax.set_ylabel ('y')
    ax.grid ()                                          # Se voglio la griglia   
    ax.legend ()

    plt.savefig ("febbraio24 (distribuzione Q2).png")
    
    array_Q2 = np.array (lista_Q2_epsilon_uniform)
    array_Q2.sort ()
    print("Soglia oltre la quale rigettare il Q2: ", array_Q2[int (N_toy * 0.9) -1])

    plt.show ()

if __name__ == "__main__" :
    main ()