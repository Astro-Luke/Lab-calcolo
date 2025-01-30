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