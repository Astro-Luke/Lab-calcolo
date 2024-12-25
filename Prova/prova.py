import numpy as np
import matplotlib.pyplot as plt
from iminuit import Minuit
from iminuit.cost import LeastSquares

from library import rand_TCL, polinomio_grad3

# ----- ----- Main ----- ----- 
def main():
    # parametri della funzione 
    a = 1.
    b = 0.7
    c = 3.
    d = -1.

    n = 10

    x = np.arange(-n, n, 1)
    y = np.zeros(2*n)

    epsilon = np.zeros(np.size(x))
    sigma = np.zeros(np.size(x))

    for i in range(x.size):
        epsilon[i] = rand_TCL(0, 3500)
        y[i] = polinomio_grad3(x[i], a, b, c, d) + epsilon[i]
    sigma = 2.5 * np.ones(np.size(y))

    least_squares = LeastSquares (x, y, sigma, polinomio_grad3)
    my_minuit = Minuit (least_squares, a = 0, b = 0, c = 0, d = 0)    # ho messo 0 come valori iniziali
    my_minuit.migrad ()                                 # minimo dei minimi quadrati
    my_minuit.hesse ()                                  

    valid = my_minuit.valid
    Q_squared = my_minuit.fval
    N_dof = my_minuit.ndof
   
    # Stampa i risultati del fit
    print("\nEsito del Fit: ", valid)
    print("\nNumero di gradi di libertà: ", N_dof)
    print("\nValore del Q-quadro: ", Q_squared, "\n")

    my_minuit.fmin
    for value, param, errore in zip (my_minuit.values, my_minuit.parameters, my_minuit.errors) : 
        print (f'{param} = {value:.6f} +/- {errore:.6f}\n')

    print("Matrice di covarianza:\n", my_minuit.covariance)

    x_fit = np.linspace (min(x), max(x), 500)
    y_fit = polinomio_grad3 (x_fit, my_minuit.values[0], my_minuit.values[1], my_minuit.values[2], my_minuit.values[3])

    # Grafico con i dati e la retta del fit
    fig, ax = plt.subplots()
    ax.set_title ("Retta con errori e fit", size = 14)
    ax.set_xlabel ("x")
    ax.set_ylabel ("y")
    ax.grid ()
    ax.errorbar (x, y, xerr = 0.0, yerr = sigma, linestyle="None", marker="o", label='Dati')
    ax.plot (x_fit, y_fit, color='red', label = 'Fit lineare')
    ax.legend ()

    plt.savefig ("prova.png") 
    plt.show ()

if __name__ == "__main__":
    main()
