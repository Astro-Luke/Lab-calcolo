import numpy as np
import random
import sys
from iminuit import Minuit
from iminuit.cost import ExtendedBinnedNLL


def sturges (N_eventi) :
    return int (np.ceil (1 + np.log2 (N_eventi)))


def rand_range (x_min, x_max) :
    return x_min + random.random() * (x_max - x_min)


def pdf_fondo (x, A = 1/2) :
    return A * np.sin (x)


#Funzione per il calcolo dell'integrale (area) e scarto secondo il metodo Hit Or Miss
def integral_HOM (f, x_min, x_max, y_min ,y_max, N_punti) :
    x_coord = []
    y_coord = []
    for _ in range (N_punti) :
        x_coord.append (rand_range (x_min, x_max))
        y_coord.append (rand_range (y_min, y_max))
    
    points_under = 0
    for x, y in zip (x_coord, y_coord) :             #zip per iterare su più liste in contemporanea
        if (f (x) > y) :
            points_under = points_under + 1
    
    A_rett = (x_max - x_min) * (y_max - y_min)
    frac = float (points_under) / float (N_punti)
    integral = A_rett * frac
    integral_incertezza = A_rett**2 * frac * (1-frac) / N_punti
    return integral, integral_incertezza


def pdf_fondo_inv () :
    y = random.random ()
    x = np.arccos (1-2*y)
    return x


def rand_TAC_gaus (mean, sigma, y_max, N = 1000) :
    normalizzazione = 1/((np.sqrt(2*np.pi))* sigma)
    x_min = mean - np.sqrt(3 * N) * sigma
    x_max = mean + np.sqrt(3 * N) * sigma
    x = rand_range (x_min, x_max)
    y = rand_range (0., y_max)
    while (y > normalizzazione * np.exp(-0.5 * ((x-mean)/sigma)**2)) :
        x = rand_range (x_min, x_max)
        y = rand_range (0., y_max)
    return x


def funzione_fit (x, mu, sigma, A) :
    return A * np.exp (-0.5 * ((x - mu)/sigma)**2)   #A = normalizzazione * numero di eventi


# Funzione per il fit con loglikelihood
def esegui_fit_LL (
        bin_content,        # contenuto dei bin
        bin_edges,          # larghezza dei bin
        dizionario_par,     # dizionario con parametri da determinare
        funzione_fit        # funzione modello da fittare
    ) :

    if not (isinstance (dizionario_par, dict)) :
        print ("Inserisci: bin_content, bin_edges, dizionario parametri e funzione da fittare.\n")
        sys.exit()

    funzione_costo = ExtendedBinnedNLL (bin_content, bin_edges, funzione_fit)
    my_minuit = Minuit (funzione_costo, **dizionario_par)
    my_minuit.migrad ()                                 
    my_minuit.hesse ()                                  

    is_valid = my_minuit.valid
    N_dof = my_minuit.ndof
    matrice_cov = my_minuit.covariance
    
    diz_risultati = {
        "Validità": is_valid,
        "Ndof": N_dof,
        "Param": my_minuit.parameters,
        "Value": my_minuit.values,
        "Errori": my_minuit.errors,
        "MatriceCovarianza": matrice_cov
    }

    return diz_risultati