# ---- ---- ---- ---- ---- Esame 15 febbraio 2023 ---- ---- ---- ---- ---- ----

# ---- ---- libreria ---- ----

import random
import numpy as np

def sturges (N_eventi) :
    return int (np.ceil (1 + np.log2 (N_eventi))) 


def rand_range (x_min, x_max) :
    return x_min + (x_max - x_min) * random.random()



def rand_TCL_unif (x_min, x_max, N) :
    y = 0. 
    for i in range (N) :
        y += rand_range (x_min, x_max)
    y /= N 
    return y 


def f (x) :
    return -((x-2)**2) + 1


def rand_TAC (f, x_min, x_max, y_max) :
    x = rand_range (x_min, x_max)
    y = rand_range (0, y_max)
    while (y > f (x)) :
        x = rand_range (x_min, x_max)
        y = rand_range (0, y_max)
    return x


def rand_TCL_para (x_min, x_max, y_max, N) :
    y = 0.
    for i in range (N) :
        y = y + rand_TAC (f, x_min, x_max, y_max)
    y /= N
    return y


# Media con lista
def media (lista) :
    mean = sum(lista)/len(lista)
    return mean

    
# Varianza con lista
def varianza (lista) :
    somma_quadrata = 0
    for elem in lista :
        somma_quadrata = somma_quadrata + (elem - media(lista))**2
    return somma_quadrata/(len(lista) -1)

    
# Deviaz. standard con lista
def dev_std (lista) :
    sigma = np.sqrt(varianza(lista))
    return sigma

    
# Deviaz. standard della media con lista
def dev_std_media (lista) :
    return dev_std(lista)/sqrt(len(lista))


# Skewness con lista
def skewness (lista):
    mean = media(lista)  # Calcola la media
    sigma = dev_std(lista)  # Calcola la deviazione standard
    n = len(lista)
    somma_cubi = 0
    for elem in lista:
        somma_cubi = somma_cubi + (elem - mean)**3
    skew = somma_cubi / (n * sigma**3)
    return skew


# Curtosi con lista
def kurtosis (lista):
    mean = media(lista)  # Calcola la media
    variance = varianza(lista)  # Calcola la varianza
    n = len(lista)
    somma_quarte = 0
    for elem in lista:
        somma_quarte = somma_quarte + (elem - mean)**4
    kurt = somma_quarte / (n * variance**2) - 3
    return kurt


# ---- ---- Main ---- ----

'''
python3 main.py
'''

import numpy as np
import matplotlib.pyplot as plt

from lib import f, rand_TCL_unif, rand_TAC, rand_TCL_para, sturges, skewness, kurtosis

def main () :

    N = 10000
    sample_TCL_unif = []
    sample_TAC = []
    sample_TCL_para = []

    for i in range (N) :
        sample_TCL_unif.append (rand_TCL_unif (0., 3., 1000))
        sample_TAC.append (rand_TAC (f, 0., 3., 1.))
        sample_TCL_para.append (rand_TCL_para (0., 3., 1., 1000))
    
    # Istogramma
    Nbin = sturges (N)

    bin_content_TCL_unif, bin_edges_TCL_unif = np.histogram (sample_TCL_unif, bins=Nbin, range = (min(sample_TCL_unif), max(sample_TCL_unif)))

    bin_content_TAC, bin_edges_TAC = np.histogram (sample_TAC, bins=Nbin, range = (min(sample_TAC), max(sample_TAC)))

    bin_content_TCL_para, bin_edges_TCL_para = np.histogram (sample_TCL_para, bins=Nbin, range = (min(sample_TCL_para), max(sample_TCL_para)))


    fig, ax = plt.subplots (nrows = 1, ncols = 3, figsize = (4, 3))
    
    ax[0].hist (sample_TCL_unif, bins=bin_edges_TCL_unif, color = 'orange')
    ax[0].set_title ('TCL uniforme', size = 14)
    ax[0].set_xlabel ('x')
    ax[0].set_ylabel ('y')
    ax[0].grid ()
    
    ax[1].hist (sample_TAC, bins=bin_edges_TAC, color = 'blue')
    ax[1].set_title ('TAC', size = 14)
    ax[1].set_xlabel ('x')
    ax[1].set_ylabel ('y')
    ax[1].grid ()

    ax[2].hist (sample_TCL_para, bins=bin_edges_TCL_para, color = 'green')
    ax[2].set_title ('TCL para', size = 14)
    ax[2].set_xlabel ('x')
    ax[2].set_ylabel ('y')
    ax[2].grid ()

    plt.savefig ('Istogrammi 15 febbraio 2023.png')
    print ("\nAsimetria a partire da distribuzione uniforme con TCL: ", skewness (sample_TCL_unif))
    print ("\nAsimetria a partire da distribuzione uniforme con TAC: ", skewness (sample_TAC))
    print ("\nAsimetria a partire da distribuzione parabolica con TCL: ", skewness (sample_TCL_para))

    print ("\nCurtosi a partire da distribuzione uniforme con TCL: ", kurtosis (sample_TCL_unif))
    print ("\nCurtosi a partire da distribuzione uniforme con TAC: ", kurtosis (sample_TAC))
    print ("\nCurtosi a partire da distribuzione parabolica con TCL: ", kurtosis (sample_TCL_para))
    
    #Punto 5
    
    N_max = 200
    lista_contatore = []

    lista_asimm_TCL_unif = []
    lista_curt_TCL_unif = []
    lista_asimm_TCL_para = []
    lista_curt_TCL_para = []
    
    for count in range (N_max) :
        asimmetria_TCL_unif = []
        curtosi_TCL_unif = []
        asimmetria_TCL_para = []
        curtosi_TCL_para = []
        
        for i in range (count+2) : 
            val_TCL_unif = rand_TCL_unif (0., 3., 10)
            val_TCL_para = rand_TCL_para (0., 3., 1., 10)
            asimmetria_TCL_unif.append (val_TCL_unif)
            curtosi_TCL_unif.append (val_TCL_unif)
            asimmetria_TCL_para.append (val_TCL_para)
            curtosi_TCL_para.append (val_TCL_para)
        
        lista_asimm_TCL_unif.append (skewness (asimmetria_TCL_unif))
        lista_curt_TCL_unif.append (kurtosis (curtosi_TCL_unif))
        lista_asimm_TCL_para.append (skewness (asimmetria_TCL_para))
        lista_curt_TCL_para.append (kurtosis (curtosi_TCL_para))

        lista_contatore.append (count)

    
    fig, ax = plt.subplots (nrows = 2, ncols = 1)
    ax[0].plot (lista_contatore, lista_asimm_TCL_unif, color = 'red', label = 'Asimmetria uniforme')
    ax[0].plot (lista_contatore, lista_asimm_TCL_para, color = 'blue', label = 'Asimmetria parabolica')
    ax[0].set_title ('TCL uniforme', size = 14)
    ax[0].set_xlabel ('x')
    ax[0].set_ylabel ('y')
    ax[0].legend ()
    ax[0].grid ()

    ax[1].plot (lista_contatore, lista_curt_TCL_unif, color = 'red', label = 'curtosi uniforme')
    ax[1].plot (lista_contatore, lista_curt_TCL_para, color = 'blue', label = 'Curtosi parabolica')
    ax[1].set_title ('TCL parabolica', size = 14)
    ax[1].set_xlabel ('x')
    ax[1].set_ylabel ('y')
    ax[1].legend ()
    ax[1].grid ()

    plt.savefig ('Andamenti asimmetria e curtosi 15 febbraio 2023.png')
    
    plt.show ()
    
if __name__ == '__main__' :
    main ()




# ---- ---- ---- ---- ---- Esame 3 luglio 2023 ---- ---- ---- ---- ---- ----

# ---- ---- Libreria ---- ----

import numpy as np
import random
import sys
from scipy.stats import chi2
from iminuit import Minuit
from iminuit.cost import LeastSquares


def sturges (N_eventi) :
    return int (np.ceil (1 + np.log2 (N_eventi))) 


def rand_range (xmin, xmax) :
    return xmin + (xmax - xmin) * random.random ()


# Funzione che genera numeri pseudocasuali partendo dal teorema centrale del limite usando media, sigma di una gaussiana
# ed N numero di eventi pseudocasuali
def rand_TCL_par_gauss (mean, sigma, N) :           # par_gauss = parametri gaussiani
    y = 0. ; 
    xMin = mean - np.sqrt(3 * N) * sigma
    xMax = mean + np.sqrt(3 * N) * sigma
    for i in range (N) :
        y += rand_range (xMin, xMax)
    y /= N 
    return y 


def funz (x) :
    return (x-2)**3 + 3


def funz_quadra (x) :
    return (x-2)**2 + 3


def funz_quadra_fit (x, a, b) :
    return (x-a)**2 + b


def funzione_fit (x, a, b) :
    return (x - a)**3 + b


def esegui_fit (
        x,                  # vettore x (np.array)
        y,                  # vettore y (np.array)
        sigma,              # vettore dei sigma (np.array)
        dizionario_par,     # dizionario con parametri 
        funzione_fit        # funzione del modello da fittare
    ) :

    if not (isinstance(dizionario_par, dict)) :
        print("Inserisci un dizionario come quarto parametro.\n")
        sys.exit()

    # Crea il modello LeastSquares
    least_squares = LeastSquares(x, y, sigma, funzione_fit)
    my_minuit = Minuit(least_squares, **dizionario_par)
    my_minuit.migrad()                                 
    my_minuit.hesse()                                  

    # Estrai i risultati principali
    is_valid = my_minuit.valid
    Q_squared = my_minuit.fval
    N_dof = my_minuit.ndof
    matrice_cov = my_minuit.covariance

    p_value = chi2.sf(Q_squared, N_dof)

    # Dizionario dei risultati
    diz_risultati = {
        "Validità": is_valid, 
        "Qsquared": Q_squared,
        "Ndof": N_dof,
        "Param": my_minuit.parameters,
        "Value": my_minuit.values,
        "Errori": my_minuit.errors,
        "MatriceCovarianza": matrice_cov,
        "Pvalue": p_value
    }

    return diz_risultati

# ---- ---- Main ---- ----

'''
python3 3luglio23.py
'''

import numpy as np
import matplotlib.pyplot as plt

from lib import funz, funzione_fit, rand_TCL_par_gauss, esegui_fit, funz_quadra, funz_quadra_fit, sturges 


def main () :
    
    x_list = [0.5, 1.5, 2.5, 3.5]
    x = np.array (x_list)
    y = np.zeros(4)
    sigma = np.zeros(4)
    
    # dizionario per la funzione esegui fit
    diz_para = {
        "a": 1., 
        "b": 1.
    }
    
    for i in range (4) :
        sigma[i] = 0.2
        y[i] = funz(x[i]) + rand_TCL_par_gauss (0., 0.2, 100)


    # Punto 2 e 3 (Grafico e Fit)
    diz_result = esegui_fit (x, y, sigma, diz_para, funzione_fit)
    
    # Per vedere i risultati del fit
    print ("\nEsito del Fit: ", diz_result["Validità"])
    print ("\nNumero di gradi di libertà: ", diz_result["Ndof"])
    print ("\nValore del Q-quadro: ", diz_result["Qsquared"])
    print ("\nP-value: ", diz_result["Pvalue"])
    print("Matrice di covarianza:\n", diz_result["MatriceCovarianza"])

    for param, value, errore in zip (diz_result["Param"], diz_result["Value"], diz_result["Errori"]) : 
        print (f'{param} = {value:.6f} +/- {errore:.6f}\n')
    
    # Grafico
    x_fit = np.linspace (0., 4., 500)
    y_fit = funzione_fit (x_fit, *diz_result["Value"])
    
    fig, ax = plt.subplots()
    ax.errorbar (x, y, xerr = 0.0, yerr = sigma, marker = 'o', linestyle = 'None', capsize = 5, capthick = 1.5, color = 'blue')
    ax.plot (x_fit, y_fit, color = 'red')
    ax.grid ()
    
    plt.savefig ("3luglio2023.png")

    # Punto 4: Toy experiment
    
    N_toy = 1000
    lista_Q2 = []
    lista_Pval = []
    
    for _ in range (N_toy) :
        sigma_toy = np.full (4, 0.2)
        y_toy = np.zeros (4)
        
        for i in range (4) :
            y_toy[i] = funz(x[i]) + rand_TCL_par_gauss (0., 0.2, 100)
        
        diz_result = esegui_fit (x, y_toy, sigma_toy, diz_para, funzione_fit)
        
        lista_Q2.append (diz_result["Qsquared"])
        lista_Pval.append (diz_result["Pvalue"])
    

    lista_quadra_Q2 = []
    lista_quadra_Pval = []
    
    for _ in range (N_toy) :
        sigma_toy = np.full (4, 0.2)
        y_toy = np.zeros (4)
        
        for i in range (4) :
            y_toy[i] = funz(x[i]) + rand_TCL_par_gauss (0., 0.2, 100)
        
        diz_result = esegui_fit (x, y_toy, sigma_toy, diz_para, funz_quadra_fit)
        
        lista_quadra_Q2.append (diz_result["Qsquared"])
        lista_quadra_Pval.append (diz_result["Pvalue"])

    
    Nbin = sturges (N_toy)
    
    bin_content_Q2, bin_edges_Q2 = np.histogram (lista_Q2, bins=Nbin, 
                                                 range = (min(lista_Q2), max(lista_Q2)))      
    bin_content_Pval, bin_edges_Pval = np.histogram (lista_Pval, bins=Nbin, 
                                                     range = (min(lista_Pval), max(lista_Pval)))

    bin_content_quadra_Q2, bin_edges_quadra_Q2 = np.histogram (lista_quadra_Q2, bins=Nbin, 
                                                               range = (min(lista_quadra_Q2), max(lista_quadra_Q2)))      
    bin_content_quadra_Pval, bin_edges_quadra_Pval = np.histogram (lista_quadra_Pval, bins=Nbin, 
                                                                   range = (min(lista_quadra_Pval), max(lista_quadra_Pval)))
    
    fig, ax = plt.subplots (nrows = 2, ncols = 2)
    ax[0][0].hist (lista_Q2, bins=bin_edges_Q2, color = 'orange', density = True)
    ax[0][0].set_title ('distribuzione Q2', size = 14)
    ax[0][0].set_xlabel ('Q2')
    ax[0][0].set_ylabel ('frequenza')
    ax[0][0].grid ()                                          # Se voglio la griglia

    ax[0][1].hist (lista_Pval, bins=bin_edges_Pval, color = 'orange', density = True)
    ax[0][1].set_title ('distribuzione P-value', size = 14)
    ax[0][1].set_xlabel ('P-value')
    ax[0][1].set_ylabel ('frequenza')
    ax[0][1].grid ()

    ax[1][0].hist (lista_quadra_Q2, bins=bin_edges_quadra_Q2, color = 'blue', density = True)
    ax[1][0].set_title ('distribuzione Q2 funz quadra', size = 14)
    ax[1][0].set_xlabel ('Q2')
    ax[1][0].set_ylabel ('frequenza')
    ax[1][0].grid ()

    ax[1][1].hist (lista_quadra_Pval, bins=bin_edges_quadra_Pval, color = 'blue', density = True)
    ax[1][1].set_title ('distribuzione P-value funz quadra', size = 14)
    ax[1][1].set_xlabel ('P-value')
    ax[1][1].set_ylabel ('frequenza')
    ax[1][1].grid ()
    
    plt.tight_layout()
    plt.savefig ('Istogrammi Q2-Pval.png')

    array_sigma = np.arange (0.1, 2., 0.1)
    lista_medie_Q2 = []
    lista_medie_quadri_Q2 = []
    for j in array_sigma :
        sigma_toy = np.full (4, j)
        list_Q2_errori = []
        list_Q2_quadri_errori = []
        
        for _ in range (N_toy) :
            y_toy = np.zeros (4)
        
            for i in range (4) :
                y_toy[i] = funz(x[i]) + rand_TCL_par_gauss (0., j, 100)
                
            diz_result = esegui_fit (x, y_toy, sigma_toy, diz_para, funzione_fit)
            diz_result_quadra = esegui_fit (x, y_toy, sigma_toy, diz_para, funz_quadra_fit)
            list_Q2_errori.append (diz_result["Qsquared"])
            list_Q2_quadri_errori.append (diz_result_quadra["Qsquared"])
            
        lista_medie_Q2.append (np.mean (list_Q2_errori))
        lista_medie_quadri_Q2.append (np.mean (list_Q2_quadri_errori))
   
    fig, ax = plt.subplots (nrows = 1, ncols = 1)
    ax.plot (array_sigma, lista_medie_Q2, label = 'fit con funz. cubica')
    ax.plot (array_sigma, lista_medie_quadri_Q2, label = 'fit con funz. quadratica')
    ax.set_title ('distribuzione medie Q2', size = 14)
    ax.set_xlabel ('sigma')
    ax.set_ylabel ('media Q2')
    ax.set_yscale ('log')
    ax.grid ()
    ax.legend ()
    
    plt.show ()

if __name__ == '__main__' :
    main ()


    
# ---- ---- ---- ---- ---- Esame 4 settembre 2023 ---- ---- ---- ---- ---- ----

# ---- ---- Libreria ---- ----
import numpy as np
import sys
import random
from iminuit import Minuit
from iminuit.cost import LeastSquares


# Funzione di controllo degli argomenti da modificare di volta in volta nel main
def controllo_arg () :
    if len (sys.argv) != 2 :       
        print("Inserire il nome del file (compresa l'estensione) ed il valore di sigma pari a 0.3 richiesto dal tema d'esame.\n")
        sys.exit()


def funzione (x) :
    return (2 * np.sin(0.5*x + 0.78) + 0.8)


def rand_range (x_min, x_max) :
    return x_min + random.random() * (x_max - x_min)


def rand_TCL_par_gauss (mean, sigma, N) :           # par_gauss = parametri gaussiani
    y = 0.
    xMin = mean - np.sqrt(3 * N) * sigma
    xMax = mean + np.sqrt(3 * N) * sigma
    for i in range (N) :
        y += rand_range (xMin, xMax)
    y /= N 
    return y 


def funzione_fit (x, p_0, p_1, p_2, p_3) :
    return (p_0 * np.sin(p_1 * x + p_2) + p_3)


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

'''
(np.sqrt (Q^2 / Ndof) ) y_0     dove y_0 è il primo valore di y supponendo che i valori di y siano tutti uguali al primo
'''

# ---- ---- Main ---- ----

'''
python3 main.py
'''
import numpy as np
import sys
import matplotlib.pyplot as plt

from lib import rand_range, funzione, rand_TCL_par_gauss, esegui_fit, funzione_fit

def main () :
    
    sigma = 0.2
    x = np.array([0.5, 2.5, 4.5, 6.5, 8.5, 10.5])
    y = np.zeros(6)
    epsilon = np.zeros(6)
    errori = []

    diz_parametri = {
        "p_0": 2.,
        "p_1": 0.5,
        "p_2": 0.78,
        "p_3": 0.8,
    }
    
    for i in range (6) :
        epsilon[i] = rand_TCL_par_gauss (0., sigma, 10)
        y[i] = funzione(x[i]) + epsilon[i]
        errori.append(sigma)
    
    np_errori = np.array(errori)

    #print ("y: \n", y, "\nerrori: \n", np_errori)

    # Parte 4: Fit
    diz_result = esegui_fit (x, y, errori, diz_parametri, funzione_fit)   
    
    print ("\nEsito del Fit: ", diz_result["Validità"])
    print ("\nNumero di gradi di libertà: ", diz_result["Qsquared"])
    print ("\nValore del Q-quadro: ", diz_result["Ndof"], "\n")
    print("Matrice di covarianza:\n", diz_result["MatriceCovarianza"])

    for param, value, errore in zip (diz_result["Param"], diz_result["Value"], diz_result["Errori"]) : 
        print (f'{param} = {value:.6f} +/- {errore:.6f}\n')

    x_fit = np.linspace (min(x), max(x), 500)
    y_fit = funzione_fit (x_fit, *diz_result["Value"])            

    fig, ax = plt.subplots ()
    ax.set_title ('Grafico e fit', size = 14)
    ax.set_xlabel ('x')
    ax.set_ylabel ('y')
    ax.errorbar (x, y, xerr = 0.0, yerr = np_errori, linestyle = 'None', marker = 'o') 
    ax.plot (x_fit, y_fit, color = 'red', label = 'Fit')
    ax.grid ()
    plt.savefig ("Settembre2023pt3.png")

    # Parte 4 e 5
    delta = np.zeros (6)
    errore_tot = np.zeros (6)
    y_new = np.zeros (6)
    
    for i in range (6) :
        delta[i] = rand_range (0, 1)
        errore_tot[i] = np.sqrt ( (epsilon[i])**2 + (delta[i])**2 )
        y_new[i] = funzione(x[i]) + errore_tot[i]

    second_diz_result = esegui_fit (x, y_new, errore_tot, diz_parametri, funzione_fit)   
    
    print ("\nEsito del Fit: ", diz_result["Validità"])
    print ("\nValore del Q-quadro: ", diz_result["Qsquared"])
    print ("\nNumero di gradi di libertà: ", diz_result["Ndof"], "\n")
    print ("Matrice di covarianza:\n", diz_result["MatriceCovarianza"])

    for param, value, errore in zip (diz_result["Param"], diz_result["Value"], diz_result["Errori"]) : 
        print (f'{param} = {value:.6f} +/- {errore:.6f}\n')

    second_x_fit = np.linspace (min(x), max(x), 500)
    second_y_fit = funzione_fit (second_x_fit, *second_diz_result["Value"])
    
    fig, ax = plt.subplots ()
    ax.set_title ('Grafico e fit dei nuovi punti', size = 14)
    ax.set_xlabel ('x')
    ax.set_ylabel ('y')
    ax.errorbar (x, y_new, xerr = 0.0, yerr = errore_tot, linestyle = 'None', marker = 'o') 
    ax.grid ()
    ax.plot (second_x_fit, second_y_fit, color = 'red', label = 'Fit con errori diversi')
    plt.savefig ("Settembre2023pt4.png")

    plt.show ()                        

if __name__ == '__main__' :
    main ()



# ---- ---- ---- ---- ---- Esame 18 settembre 2023 ---- ---- ---- ---- ---- ----

# ---- ---- Libreria ---- ----

import numpy as np
import random

# Funzione sturges per il binnaggio (funziona discretamente bene, ma conviene sempre veerificare)
def sturges (N_eventi) :
    return int (np.ceil (1 + np.log2 (N_eventi))) 


def rand_range (x_min, x_max) :
    return x_min + (x_max - x_min) * random.random ()


def f_cauchy (M, gamma) :
    x_min = M - 3. * gamma
    x_max = M + 3. * gamma
    ymax = 1.
    x = rand_range (x_min, x_max)
    y = rand_range (0., ymax)
    while (y > (1/np.pi) * (gamma/(x-M)**2 + gamma**2)) :
        x = rand_range (x_min, x_max)
        y = rand_range (0., ymax)
    return x

def generate_cauchy (M, gamma) :
    lista_cauchy = []
    lista_count = []
    medie_cauchy = []
    std_cauchy = []
    ymax = 1.
    for i in np.arange (1, 101, 1) :
        x_min = M - i*gamma
        x_max = M + i*gamma
        x = rand_range (x_min, x_max)
        y = rand_range (0., ymax)
        while (y > (1/np.pi) * (gamma/(x-M)**2 + gamma**2)) :
            x = rand_range (x_min, x_max)
            y = rand_range (0., ymax)
        lista_cauchy.append (x)
        lista_count.append (i)
        medie_cauchy.append (np.mean(lista_cauchy))
        std_cauchy.append (np.std(lista_cauchy))
    return medie_cauchy, std_cauchy, lista_count


def rand_TCL_cauchy (gamma, M, N = 100) :
    y = 0.
    for i in range (N) :
        y = y + f_cauchy (M, gamma)
    y /= N 
    return y


# ---- ---- Main ---- ----

'''
python3 main.py
'''

import numpy as np
import matplotlib.pyplot as plt

from lib import sturges, f_cauchy, generate_cauchy, rand_TCL_cauchy

def main () :

    # Punto 1 e 2
    gamma = 1.
    M = 0.5
    N = 1000

    casual_cauchy = []
    for i in range (N) :
        casual_cauchy.append (f_cauchy (gamma, M))

    Nbin = sturges (N)
    bin_content, bin_edges = np.histogram (casual_cauchy, bins=Nbin, range = (min(casual_cauchy), max(casual_cauchy)))

    fig, ax = plt.subplots ()
    ax.hist (casual_cauchy, bins = Nbin, color = 'orange', label = 'f_cauchy')
    ax.set_title('Istogramma distribuzione Cauchy')
    ax.set_xlabel ('x')
    ax.set_ylabel ('y')
    ax.grid ()
    ax.legend ()

    # Punto 4
    lista_medie_cauchy, std_cauchy, lista_contatore = generate_cauchy (M, gamma)

    fig, ax = plt.subplots ()
    ax.plot (lista_contatore, lista_medie_cauchy, color = 'blue', label = 'Media')
    ax.plot (lista_contatore, std_cauchy, color = 'red', label = 'Sigma')
    ax.set_title('Andamento della media e dev_std')
    ax.set_xlabel ('i')
    ax.set_ylabel ('media e sigma')
    ax.grid ()
    ax.legend ()

    # Punto 5
    N_pt5 = 10000
    lista_TCL_cauchy = []
    for i in range (N) :
        lista_TCL_cauchy.append (rand_TCL_cauchy (gamma, M))

    Nbin = sturges (N_pt5)
    #print (lista_TCL_cauchy)
    bin_content, bin_edges = np.histogram (lista_TCL_cauchy, bins=Nbin, range = (min (lista_TCL_cauchy), max (lista_TCL_cauchy)) )
    
    fig, ax = plt.subplots ()
    ax.hist (lista_TCL_cauchy, bins=bin_edges, color = 'orange')
    ax.set_title ('Distribuzione Cauchy con TCL')
    ax.set_xlabel ('x')
    ax.set_ylabel ('y')
    ax.grid ()
    
    plt.show ()

if __name__ == '__main__' :
    main ()
