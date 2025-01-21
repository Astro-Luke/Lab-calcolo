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