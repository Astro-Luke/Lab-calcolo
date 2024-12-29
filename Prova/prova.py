'''
Ripetere l'esercizio di adattamento per un andamento parabolico.
'''
import numpy as np
import matplotlib.pyplot as plt
from iminuit import Minuit
from iminuit.cost import LeastSquares

from library import rand_TCL_par_gauss, polinomio_grad3

def main () :

    a = 0.3
    b = 0.7
    c = -1.
    d = 2.

    x = np.arange(-10, 10, 1)
    y = np.zeros(x.size)

    epsilon = np.zeros(np.size(x))
    sigma_barra = np.zeros (np.size(x))

    for i in range (x.size) :
        epsilon[i] = rand_TCL_par_gauss (0, 3, 1000)                   # 10 era il valore iniziale incriminato, troppo piccolo. 100 mi permette di distribuire meglio i punti
        y[i] = polinomio_grad3 (x[i], a, b, c, d) + epsilon[i]
    
    sigma_barra = 3. * np.ones(np.size (y))
    
    least_squares = LeastSquares (x, y, sigma_barra, polinomio_grad3)
    my_minuit = Minuit (least_squares, a = 0., b = 0., c = 0., d = 0.)    # ho messo 0 come valori iniziali
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

    # Calcola la parabola del fit
    x_fit = np.linspace (min(x), max(x), 500)
    y_fit = polinomio_grad3 (x_fit, my_minuit.values[0], my_minuit.values[1], my_minuit.values[2], my_minuit.values[3])

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