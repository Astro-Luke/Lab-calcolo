import numpy as np
from iminuit import Minuit
from iminuit.cost import LeastSquares
import random
import sys

# Funzione Hubble
def legge_hubble (redshift, H) :
    c = 3 * (10)**5
    D = (redshift * c) / (H)
    return D


def accelerazione_uni (redshift, H, omega) :
    c = 3 * (10)**5
    q = ((3 * omega) / 2 ) - 1
    D = (c/H) * (redshift + 0.5 * (1 - q) * (redshift)**2)
    return D


# Funzione che esegue il fit 
def esegui_fit (x, y, sigma, dizionario_par, funzione_fit) :

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


# Distribuzione uniforme tra x_min e x_max con seed scelto in auto
def rand_range (x_min, x_max) :
    return x_min + random.random() * (x_max - x_min)


def leggi_file_dati (nome_file):
    '''
    Legge un file di dati con valori separati da spazi e lo converte in un array NumPy.
    Argomenti: nome del file (ad esempio mettendo nome_file = "SuperNovae.txt") assicurandosi che sia nella stessa directory
    Return: tuple, un array NumPy con i dati e il numero di righe del file.
    '''
    with open (nome_file, 'r') as file:
        lines = file.readlines()
        lista_dati = []
        
        for line in lines:
            lista_string = line.split()
            list_float = [float(x) for x in lista_string]
            lista_dati.append(list_float)
        
        sample = np.array(lista_dati)
        N_righe = len(sample)
    
    return sample, N_righe
