'''
linea di comando: python3 Hubble.py
'''

# ----- Librerie -----

import matplotlib.pyplot as plt
import numpy as np

from lib import esegui_fit, legge_hubble, accelerazione_uni, rand_range, leggi_file_dati

# Lib

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



# ----- Main -----

def main () :
    
    # Parte 1
    redshift, distanza, errore = np.loadtxt ("SuperNovae.txt", unpack = True)     # unpack in questo caso è necessario perchè ho più colonne
    
    diz_par_lin = {         # non ha molto senso fare un diz. con un solo parametro, però non volevo modificare la funzione esegui_fit
        "H": 1.,            # questo è il dizionario per la legge lineare
    }

    # Parte 2 e 3
    diz_result = esegui_fit (redshift, distanza, errore, diz_par_lin, legge_hubble)     # fit con legge lineare

    print ("\nEsito del Fit: ", diz_result["Validità"])
    print ("\nNumero di gradi di libertà: ", diz_result["Ndof"])
    print ("\nValore del Q-quadro: ", diz_result["Qsquared"], "\n")

    print("Matrice di covarianza:\n", diz_result["MatriceCovarianza"])

    for param, value, errore in zip (diz_result["Param"], diz_result["Value"], diz_result["Errori"]) : 
        print (f'{param} = {value:.6f} +/- {errore:.6f}\n')

    x_fit = np.linspace (min(redshift), max(redshift), 500)     # creo un array per l'asse x del grafico
    y_fit = legge_hubble (x_fit, *diz_result["Value"])          # calcolo le y da mettere nel grafico 
    

    # Parte 4
    diz_par_acc = {
        "H": 1.,
        "omega": 1.
    }

    diz_result_acc = esegui_fit (redshift, distanza, errore, diz_par_acc, accelerazione_uni)        # fit con funzione accelerata

    print ("\nEsito del Fit: ", diz_result_acc["Validità"])
    print ("\nNumero di gradi di libertà: ", diz_result_acc["Qsquared"])
    print ("\nValore del Q-quadro: ", diz_result_acc["Ndof"], "\n")

    print("Matrice di covarianza:\n", diz_result_acc["MatriceCovarianza"])

    for param, value, errore in zip (diz_result_acc["Param"], diz_result_acc["Value"], diz_result_acc["Errori"]) : 
        print (f'{param} = {value:.6f} +/- {errore:.6f}\n')

    x_fit_acc = np.linspace (min(redshift), max(redshift), 500)             # asse delle x da mettere nel grafico
    y_fit_acc = accelerazione_uni (x_fit, *diz_result_acc["Value"])         # calcolo le y da mettere nel grafico con la funzione accelerata
    
    fig, ax = plt.subplots(nrows = 1, ncols = 1, figsize = (10, 5))   # 1 riga, 1 colonna
    ax.errorbar (redshift, distanza, xerr = 0.0, yerr = errore,
        markersize = 3,                             # dimensione del punto
        fmt = 'o',                                  # tipo di marker (punto)
        color = 'blue',                             # colore della linea
        ecolor = 'red',                             # colore della barra di error
        )
    
    # Creazione del grafico
    ax.plot (x_fit, y_fit, color = 'red', label = 'Fit lineare')
    ax.plot (x_fit_acc, y_fit_acc, color = 'green', label = "Fit accelerato")
    ax.grid ()
    ax.set_title ("Plot dei dati con legge lineare e accelerata")
    ax.legend ()
    ax.set_xlabel ("Redshift")
    ax.set_ylabel ("Distanza (Mpc)")

    plt.savefig ("Plot_e_fit_Hubble.png")
    
    # Parte 5
    sub_sample, N_righe = leggi_file_dati ("SuperNovae.txt")
    
    righe_selezionate = []
    N = 30                      # numero di righe da estrarre
    for i in range (N) :
        indice = int (rand_range (0, N_righe -1))
        righe_selezionate.append (sub_sample[indice])     # qui serve un bel casting obbligatorio!

    righe_selezionate = np.array (righe_selezionate)
    # print ("\n", righe_selezionate, "\n")     # ora è solo di controllo questa riga
    
    redshift_subsample = righe_selezionate[:, 0]             # Slicing, estrae tutti gli elementi della colonna 1.
    distanza_subsample = righe_selezionate[:, 1]
    errore_subsample = righe_selezionate[:, 2]

    diz_result_subsample = esegui_fit (redshift_subsample, distanza_subsample, errore_subsample, diz_par_acc, accelerazione_uni)

    print ("\nEsito del Fit: ", diz_result_subsample["Validità"])
    print ("\nNumero di gradi di libertà: ", diz_result_subsample["Qsquared"])
    print ("\nValore del Q-quadro: ", diz_result_subsample["Ndof"], "\n")

    print("Matrice di covarianza:\n", diz_result_subsample["MatriceCovarianza"])

    for param, value, errore in zip (diz_result_subsample["Param"], diz_result_subsample["Value"], diz_result_subsample["Errori"]) : 
        print (f'{param} = {value:.6f} +/- {errore:.6f}\n')

    x_fit_subsample = np.linspace (min(redshift_subsample), max(redshift_subsample), 500)             # asse delle x da mettere nel grafico
    y_fit_subsample = accelerazione_uni (x_fit_subsample, *diz_result_subsample["Value"])         # calcolo le y da mettere nel grafico con la funzione accelerata

    fig, ax = plt.subplots (nrows = 1, ncols = 1, figsize = (10, 5))   # 1 riga, 1 colonna
    ax.errorbar (redshift_subsample, distanza_subsample, xerr = 0.0, yerr = errore_subsample,
        markersize = 3,                             # dimensione del punto
        fmt = 'o',                                  # tipo di marker (punto)
        color = 'blue',                             # colore della linea
        ecolor = 'red',                             # colore della barra di error
        )
    
    # Creazione del grafico
    ax.plot (x_fit_subsample, y_fit_subsample, color = 'green', label = 'Fit con sottocampione')
    ax.grid ()
    ax.set_title ("Plot dei dati con sottocampione casuale")
    ax.legend ()
    ax.set_xlabel ("Redshift")
    ax.set_ylabel ("Distanza (Mpc)")

    plt.savefig ("Plot casuale Hubble.png")

    plt.show ()  

if __name__ == "__main__" :
    main ()
