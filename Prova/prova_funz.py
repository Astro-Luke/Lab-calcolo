'''
Ripetere l'esercizio di adattamento per un andamento parabolico.
'''
import numpy as np
import matplotlib.pyplot as plt
from iminuit import Minuit
from iminuit.cost import LeastSquares

from library import rand_TCL_par_gauss, polinomio_grad3, esegui_fit

def main () :
    
    dizionario = {
        "a": 0.3,
        "b": 0.7,
        "c": -1.,
        "d": 2.
    }

    x = np.arange(-10, 10, 1)
    y = np.zeros(x.size)

    epsilon = np.zeros(np.size(x))
    sigma_barra = np.zeros (np.size(x))

    for i in range (x.size) :
        epsilon[i] = rand_TCL_par_gauss (0, 3, 1000)                   # 10 era il valore iniziale incriminato, troppo piccolo. 100 mi permette di distribuire meglio i punti
        y[i] = polinomio_grad3 (x[i], **dizionario) + epsilon[i]
    
    sigma_barra = 3. * np.ones(np.size (y))
    
    diz_result = esegui_fit (x, y, sigma_barra, dizionario, polinomio_grad3)

    '''
    for value, param, errore in zip (my_minuit.values, my_minuit.parameters, my_minuit.errors) : 
        print (f'{param} = {value:.6f} +/- {errore:.6f}\n')

    '''

    print ("\nEsito del Fit: ", diz_result["Validità"])
    print ("\nNumero di gradi di libertà: ", diz_result["Qsquared"])
    print ("\nValore del Q-quadro: ", diz_result["Ndof"], "\n")

    print("Matrice di covarianza:\n", diz_result["MatriceCovarianza"])

    for param, value, errore in zip (diz_result["Param"], diz_result["Value"], diz_result["Errori"]) : 
        print (f'{param} = {value:.6f} +/- {errore:.6f}\n')

    # Calcola la parabola del fit
    x_fit = np.linspace (min(x), max(x), 500)
    y_fit = polinomio_grad3 (x_fit, *diz_result["Value"])

    # Grafico con i dati e la retta del fit
    fig, ax = plt.subplots()
    ax.set_title ("Parabola con errori e fit", size = 14)
    ax.set_xlabel ("x")
    ax.set_ylabel ("y")
    ax.grid ()
    ax.errorbar (x, y, xerr = 0.0, yerr = 3., linestyle="None", marker="o", ecolor = 'green', elinewidth = 1.5, capsize = 2.5, capthick = 1.5, label='Dati')
    ax.plot (x_fit, y_fit, color='red', label = 'Fit lineare')
    ax.legend ()

    plt.savefig ("prova.png")    
    plt.show ()

if __name__ == "__main__" :
    main ()