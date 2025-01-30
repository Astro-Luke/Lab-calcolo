# ---- ---- ---- ---- Esame 22 gennaio 2024 ---- ---- ---- ---- 

# ---- ---- Libreria ---- ----
import random
import numpy as np
from math import ceil, sqrt, pow

# Funzione sturges per il binnaggio (funziona discretamente bene, ma conviene sempre veerificare)
def sturges (N_eventi) :
    return ceil (1 + np.log2 (N_eventi))

# Distribuzione uniforme tra x_min e x_max con seed scelto in auto
def rand_range (x_min, x_max) :
    return x_min + random.random() * (x_max - x_min)

def funzione_per_normalizzare (x) :
    return (np.cos(x))**2

#Funzione per il calcolo dell'integrale (area) e scarto secondo il metodo Hit Or Miss
def integral_HOM (f, x_min, x_max, y_min, y_max, N_punti) :
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


def funzione_globale (x) :
    A = 0.
    val_int, incert_integ = integral_HOM(funzione_per_normalizzare, 0, (3/2)*np.pi, 0, 1, 1000)
    A = A + (val_int**(-1))
    if (x > 0 and x < ((3/2) * np.pi)) :
        return A*(np.cos(x))**2
    else :
        return 0.

#Funzione che genera numeri pseudocasuali tramite l'argoritmo Try And Catch e distribuzione uniforme rand_range
def rand_TAC (f, x_min, x_max, y_max) :
    x = rand_range (x_min, x_max)
    y = rand_range (0, y_max)
    while (y > f (x)) :
        x = rand_range (x_min, x_max)
        y = rand_range (0, y_max)
    return x

# Media con lista
def media (lista) :
    mean = sum(lista)/len(lista)
    return mean

# Varianza con lista
def varianza (lista) :
    somma_quadrata = 0
    for elem in lista :
        somma_quadrata = somma_quadrata + (elem - media(lista))**2
    return somma_quadrata/(len(lista))

# Deviaz. standard con lista
def dev_std (lista) :
    sigma = sqrt(varianza(lista))
    return sigma

# Skewness con lista
def skewness(lista):
    mean = media(lista)  # Calcola la media
    sigma = dev_std(lista)  # Calcola la deviazione standard
    n = len(lista)
    somma_cubi = 0
    for elem in lista:
        somma_cubi = somma_cubi + (elem - mean)**3
    skew = somma_cubi / (n * sigma**3)
    return skew

# Curtosi con lista
def kurtosis(lista):
    mean = media(lista)  # Calcola la media
    variance = varianza(lista)  # Calcola la varianza
    n = len(lista)
    somma_quarte = 0
    for elem in lista:
        somma_quarte = somma_quarte + (elem - mean)**4
    kurt = somma_quarte / (n * variance**2) - 3
    return kurt

# Funzione che genera numeri pseudocasuali partendo dal teorema centrale del limite
def rand_TCL (xMin, xMax, N = 1000) :
    y = 0.
    for _ in range (N) :
        y = y + rand_range (xMin, xMax)
    y /= N
    return y


# ---- ---- Main ----- -----

'''
python3 gennaio24.py
'''

import matplotlib.pyplot as plt
import numpy as np
from lib import integral_HOM, funzione_per_normalizzare, rand_TAC, sturges, funzione_globale, media, dev_std, skewness, kurtosis, rand_TCL

def main () :

    x_min = 0
    x_max = (3/2) * np.pi
    val_int, integral_incertezza = integral_HOM (funzione_per_normalizzare, x_min, x_max, 0, 1, 10000)

    print ("Valore integrale: ", val_int, "+/-", integral_incertezza)

    A = (val_int**(-1))
    print ("Costante di normalizzazione A: ", A)
    print ("Area normalizzata: ", A * val_int)

    N = 10000
    lista_casuali = []
    for _ in range (N) :
        lista_casuali.append (rand_TAC (funzione_globale, x_min, x_max, 1))

    Nbin = sturges (N) + 5

    bin_edges = np.linspace (x_min, x_max, Nbin)         # Regola la dimensione dei bin e Nbin = numero di bin
    fig, ax = plt.subplots (nrows = 1, ncols = 1)
    ax.hist (lista_casuali, bins=bin_edges, color = 'orange')
    ax.set_xlabel ('x')
    ax.set_ylabel ('y')
    ax.grid ()                                         
    
    print ("\nLa media è: ", media(lista_casuali))
    print ("\nLa deviazione standard è: ", dev_std(lista_casuali))

    print ("\nLa asimmetria è: ", skewness(lista_casuali))
    print ("\nLa curtosi è: ", kurtosis(lista_casuali), "\n")

    '''
    list_TCL = []
    for _ in range (N) :
        list_TCL.append (rand_TCL (x_min, x_max))
    '''
    
    plt.savefig ('Istogramma gennaio24.png')
    plt.show ()    

if __name__ == "__main__" :
    main ()



# ---- ---- ---- ---- Esame 5 febbraio 2024 ---- ---- ---- ---- 

# ---- ---- Libreria ---- ----

import numpy as np
import random
import sys
from math import ceil
from iminuit import Minuit
from iminuit.cost import LeastSquares

def parabola (x, a, b, c) :
    return a + b*x + c*(x**2)


# Funzione sturges per il binnaggio (funziona discretamente bene, ma conviene sempre veerificare)
def sturges (N_eventi) :
    return ceil (1 + np.log2 (N_eventi))


# Distribuzione uniforme tra x_min e x_max con seed scelto in auto
def rand_range (x_min, x_max) :
    return x_min + random.random() * (x_max - x_min)


# Funzione che genera numeri pseudocasuali partendo dal teorema centrale del limite usando media, sigma di una gaussiana
# ed N numero di eventi pseudocasuali
def rand_TCL_par_gauss (mean, sigma, N) :           # par_gauss = parametri gaussiani
    y = 0. 
    xMin = mean - np.sqrt(3 * N) * sigma
    xMax = mean + np.sqrt(3 * N) * sigma
    for i in range (N) :
        y = y + rand_range (xMin, xMax)
    y /= N 
    return y 


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


# ---- ---- Main ---- ----

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


# ---- ---- ---- ---- Esame 19 febbraio 2024 ---- ---- ---- ---- 

# ---- ---- Libreria ---- ----

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

# ---- ---- Main ---- ----

'''
linea di comando: python3 Hubble.py
'''

# ----- Librerie -----

import matplotlib.pyplot as plt
import numpy as np

from lib import esegui_fit, legge_hubble, accelerazione_uni, rand_range, leggi_file_dati

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



# ---- ---- ---- ---- Esame 24 giugno 2024 ---- ---- ---- ---- 

# ---- ---- Libreria ---- ----
import numpy as np
import random
from scipy.stats import expon, norm
from iminuit import Minuit
from iminuit.cost import ExtendedBinnedNLL
import sys

# Funzione sturges per il binnaggio (funziona discretamente bene, ma conviene sempre veerificare)
def sturges (N_eventi) :
    return int (np.ceil (1 + np.log2 (N_eventi)))


# Distribuzione uniforme tra x_min e x_max con seed scelto in auto
def rand_range (x_min, x_max) :
    return x_min + random.random() * (x_max - x_min)


def rand_exp_inversa (t) :
    return -1. * np.log (1 - random.random()) * t


def mod_total (bin_edges, N_signal, mu_function, sigma_function, N_background, tau_function) :
    return N_signal * norm.cdf (bin_edges, mu_function, sigma_function) + N_background * expon.cdf (bin_edges, 0, tau_function)


def pdf_tot (x, a, mu, sigma, b, tau) :
    lambd = 1/tau
    return a * lambd * np.exp(-lambd * x) + b * (1/(np.sqrt(2 * np.pi) * sigma)) * np.exp(-0.5 * (((x - mu) / sigma))**2 )


def rand_TAC_exp (lambd, N) :   # N numero di num pseudocas da generare
    sample = []
    y_max = lambd
    tau = 1/lambd
    x_max = 3 * tau
    for i in range (N) :
        x = rand_range (0., x_max)
        y = rand_range (0., y_max)      # pomgo lambd come y_max 
        while (y > lambd * (np.exp (- x * lambd))) :
            x = rand_range (0., x_max)
            y = rand_range (0., y_max)
        sample.append (x)
    return (sample)


def rand_TAC_gaus (mu, sigma, N) :
    sample = []
    y_max = 1.
    for i in range (N) :
        x = rand_range (mu - 3. * sigma, mu + 3. * sigma)
        y = rand_range (0., y_max)
        while (y > np.exp (-0.5 * ( ((x - mu) / sigma)**2) ) ) :
            x = rand_range (mu - 3. * sigma, mu + 3. * sigma)
            y = rand_range (0., y_max)
        sample.append (x)
    return sample


def rand_TCL_par_gauss (mean, sigma, N) :           # par_gauss = parametri gaussiani
    y = 0. 
    xMin = mean - np.sqrt(3 * N) * sigma
    xMax = mean + np.sqrt(3 * N) * sigma
    for i in range (N) :
        y += rand_range (xMin, xMax)
    y /= N 
    return y 


# Funzione per il calcolo del massimo della loglikelihood
def sezioneAureaMax_LL (
    f,              # funzione di likelihood da massimizzare
    pdf,            # probability density function degli eventi
    sample,         # campione degli eventi
    x0,             # estremo dell'intervallo
    x1,             # altro estremo dell'intervallo
    prec=0.0001     # precisione della funzione
) :
    r = 0.618  # Costante aurea
    x2 = 0.
    x3 = 0.
    larghezza = abs(x1 - x0)

    while larghezza > prec:
        x2 = x0 + r * (x1 - x0)
        x3 = x0 + (1. - r) * (x1 - x0)

        # Restringimento dell'intervallo
        if f(sample, pdf, x3) < f(sample, pdf, x2) :
            x0 = x3
        else:
            x1 = x2
        larghezza = abs(x1 - x0)

    return (x0 + x1) / 2.   # Ritorna il punto medio dell'intervallo finale


#Funzione likelihood logaritmica
def loglikelihood (an_array, pdf, para) :
    result = 0.
    for x in an_array :
        val_pdf = pdf (x, *para)
        if val_pdf > 0. :
            result = result + np.log (val_pdf)
    return result


def esegui_fit_LL (bin_content, bin_edges, dizionario_par, funzione_fit) :

    if not (isinstance (dizionario_par, dict)) :
        print ("Inserisci un dizionario come secondo parametro.\n")
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


# ---- ---- Main ---- ----

'''
python3 main.py
'''

import numpy as np
import matplotlib.pyplot as plt
from iminuit.cost import ExtendedBinnedNLL
from iminuit import Minuit
from tabulate import tabulate

from lib import sturges, rand_exp_inversa, rand_TCL_par_gauss, mod_total, loglikelihood, pdf_tot, esegui_fit_LL

# Funzione per il calcolo del massimo della loglikelihood
def sezioneAureaMax_LL(
    f,              # funzione di likelihood da massimizzare
    pdf,            # probability density function degli eventi
    sample,         # campione degli eventi
    x0,             # estremo dell'intervallo
    x1,             # altro estremo dell'intervallo
    prec=0.0001,    # precisione della funzione
    params=None     # parametri aggiuntivi per la funzione f
):
    r = 0.618  # Costante aurea
    x2 = 0.
    x3 = 0.
    larghezza = abs(x1 - x0)

    if params is None:
        params = []

    while larghezza > prec:
        x2 = x0 + r * (x1 - x0)
        x3 = x0 + (1. - r) * (x1 - x0)

        # Restringimento dell'intervallo
        if f(sample, pdf, params + [x3]) < f(sample, pdf, params + [x2]):
            x0 = x3
        else:
            x1 = x2
        larghezza = abs(x1 - x0)

    return (x0 + x1) / 2.   # Ritorna il punto medio dell'intervallo finale


def main():
    # Punto 1
    N_exp = 2000
    tau = 200
    lambd = (1 / tau)
    campione_exp = [rand_exp_inversa(tau) for _ in range(N_exp)]

    N_gau = 200
    mu = 190
    sigma = 20
    campione_gauss = [rand_TCL_par_gauss(mu, sigma, 10000) for _ in range(N_gau)]

    # Punto 2
    campione_totale = campione_gauss + campione_exp
    
    Nbin = int(sturges(len(campione_totale))) + 15
    bin_content, bin_edges = np.histogram(campione_totale, bins=Nbin, range=(0., 3 * tau))

    fig, ax = plt.subplots()
    ax.hist(campione_totale, bins=bin_edges, color='orange')
    ax.grid()
    plt.savefig('Istogramma_esame_giugno24.png')

    # Punto 3: Fit
    media_campione = np.mean(campione_totale)
    sigma_campione = np.std(campione_totale)
    N_eventi = np.sum(bin_content)

    diz_variabili = {
        "N_signal": N_eventi,
        "mu_function": media_campione,
        "sigma_function": sigma_campione,
        "N_background": N_eventi,
        "tau_function": tau
    }

    diz_result = esegui_fit_LL(bin_content, bin_edges, diz_variabili, mod_total)

    for param, value, errore in zip(diz_result["Param"], diz_result["Value"], diz_result["Errori"]):
        print(f'{param} = {value:.6f} +/- {errore:.6f}\n')

    # Punto 4
    valori_parametri = list(diz_result["Value"])
    log_value = loglikelihood(campione_totale, pdf_tot, valori_parametri)
    print("Valore della loglikelihood:", log_value)

    valori_loglike = []
    for mu in np.arange(30, 300, 0.5):
        valori_parametri[1] = mu  # Aggiorna solo il parametro mu
        valori_loglike.append(loglikelihood(campione_totale, pdf_tot, valori_parametri))

    # Punto 5
    maximum = sezioneAureaMax_LL(loglikelihood, pdf_tot, campione_totale, 120, 250, params=valori_parametri[:-1])
    print("Massimo della loglikelihood trovato a mu:", maximum)

    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(30, 300, 0.5)
    ax.plot(x, valori_loglike, label="loglikelihood")
    ax.legend()
    ax.grid()
    ax.set_title("Plot loglikelihood al variare di mu con massimo evidenziato")
    plt.savefig("Plot_loglikelihood_con_massimo.png")

    plt.show()

if __name__ == "__main__":
    main ()



# ---- ---- ---- ---- Esame 8 luglio 2024 ---- ---- ---- ---- 

# ---- ---- Libreria ---- ----
import numpy as np
import random
from math import ceil
from scipy.stats import rayleigh

def sturges (N_eventi) :
    return int (ceil (1 + np.log2 (N_eventi))) 


# Distribuzione uniforme tra x_min e x_max con seed scelto in auto
def rand_range (x_min, x_max) :
    return x_min + random.random() * (x_max - x_min)


#Funzione che genera numeri pseudocasuali gaussiani con TAC
def rand_TAC_gaus (mu, sigma) :
    y_max = 1.
    if (mu - 3. * sigma) < 0 :
        x_sx = 0.
    else :
        x_sx = mu - 3. * sigma
    x = rand_range (x_sx, mu + 3. * sigma)
    y = rand_range (0., y_max)
    while (y > np.exp (-0.5 * ( ((x - mu) / sigma)**2) ) ) :
        x = rand_range (x_sx, mu + 3. * sigma)
        y = rand_range (0., y_max)
    return x


def random_walk (mean, sigma, N_passi) :
    asse_x = [0.]
    asse_y = [0.]
    for _ in range (N_passi) : 
        theta = rand_range (0., 2*np.pi)
        ro = rand_TAC_gaus (mean, sigma)
        x = asse_x[-1] + ro * np.cos (theta)
        y = asse_y[-1] + ro * np.sin (theta)
        asse_x.append (x)
        asse_y.append (y)
    return asse_x, asse_y


def calcola_distanza (x_0, x_n, y_0, y_n) :
    return np.sqrt( ((x_n - x_0)**2) + ((y_n - y_0)**2) )


def funzione_fit (bin_edges, N) :
    return rayleigh.cdf (bin_edges, loc = 0, scale = np.sqrt (N/2))


def Rayleigh (r, N) :
    return ((2*r)/N) * np.exp(-(r**2)/N)


# -------------- Statistiche --------------

# Media con array
def media (sample) :
    mean = np.sum(sample)/len(sample)
    return mean
    

# Varianza con array
def varianza (sample) :
    somma_quadrata = 0
    somma_quadrata = np.sum( (sample - media (sample))**2 )
    var = somma_quadrata/(len (sample))
    return var


# Varianza con corr. di Bessel con array
def varianza_bessel (sample) :
    somma_quadrata = 0
    somma_quadrata = np.sum( (sample - media(sample))**2 )
    var = somma_quadrata/(len(sample) - 1)
    return var


# Deviaz. standard con array
def dev_std (sample) :
    sigma = np.sqrt (varianza(sample))
    return sigma


# Deviaz. standard della media con array
def dev_std_media (sample) :
    return dev_std(sample) / (np.sqrt( len(sample) ))


# Skewness con array
def skewness (sample) :
    mean = media (sample)  # Calcola la media con la tua funzione
    sigma = dev_std (sample)  # Calcola la deviazione standard con la tua funzione
    n = len(sample)
    skew = np.sum((sample - mean)**3) / (n * sigma**3)
    return skew


# Curtosi con array
def kurtosis (sample) :
    mean = media (sample)  # Calcola la media con la tua funzione
    variance = varianza (sample)  # Calcola la varianza con la tua funzione
    n = len(sample)
    kurt = np.sum((sample - mean)**4) / (n * variance**2) - 3
    return kurt


# ---- ---- Main ---- ----

'''
python3 luglio2024.py
'''
import numpy as np
import matplotlib.pyplot as plt
from iminuit import Minuit
from iminuit.cost import ExtendedBinnedNLL
from scipy.stats import rayleigh
from lib import random_walk, calcola_distanza, media, varianza_bessel, skewness, kurtosis, sturges, funzione_fit, Rayleigh

def main () :

    # Punto 1 e 2
    mean = 1.
    sigma = 0.2
    N_passi = 10
    coord_x, coord_y = random_walk (mean, sigma, N_passi)
    
    #print ("\n", coord_x, "\n")
    #print (coord_y)
    
    # calcolo la distanza tra il punto (x, y) = (0, 0) ed il punto raggiunto
    coord_x_array = np.array (coord_x)
    coord_y_array = np.array (coord_y)

    distanza = calcola_distanza (0., coord_x_array[10], 0., coord_y_array[10])
    print ("La distanza dal punto di partenza (0, 0) al punto", "(x, y) = (", coord_x_array[10], ",", 
           coord_y_array[10], ") è: \n", distanza, "\n")

    # Grafico
    fig, ax = plt.subplots ()
    ax.plot (coord_x, coord_y, "o-", color = "blue")
    ax.set_xlabel ("x")
    ax.set_ylabel ("y")
    ax.grid ()
    plt.savefig ("Grafico ubriaco.png")

    # Punto 3
    N_persone = 10000
    list_distanze = []
    for _ in range (N_persone) :
        coord_x, coord_y = random_walk (mean, sigma, N_passi)
        distanza = calcola_distanza (0., coord_x[10], 0., coord_y[10])
        list_distanze.append (distanza)


    Nbin = sturges (len (list_distanze))

    bin_content, bin_edges = np.histogram (list_distanze, bins=Nbin, range = (min (list_distanze), max(list_distanze)))
    #print (bin_edges)
    '''
    fig, ax = plt.subplots (nrows = 1, ncols = 1)
    ax.hist (list_distanze, bins = bin_edges, color = 'orange')
    ax.set_title ('Plot persone diversamente sobrie', size = 14)
    ax.set_xlabel ('x')
    ax.set_ylabel ('Distanza percorsa')
    ax.grid ()                                        
    plt.savefig ('Persone ubriache.png')
    '''
    # Punto 4
    vett_distanza = np.array (list_distanze)    # casting

    print ("\n----- Statistiche della distribuzione -----\n\nMedia: ", media (list_distanze))
    print ("\nVarianza: ", varianza_bessel (list_distanze))
    print ("\nAsimmetria: ", skewness (list_distanze))
    print ("\nCurtosi: ", kurtosis (list_distanze))

    # punto 5: Fit
    N_passi = 10
    funz_costo = ExtendedBinnedNLL (bin_content, bin_edges, funzione_fit)
    my_minuit = Minuit (funz_costo, N_passi)
    my_minuit.migrad ()

    print ("Esito del Fit: ", my_minuit.valid)
    print ("Valore: ", my_minuit.values[0], "+/-", my_minuit.errors[0])


    fig, ax = plt.subplots (nrows = 1, ncols = 1)
    x = np.linspace (min (bin_edges), max (bin_edges), 500)

    ax.hist (list_distanze, bins = bin_edges, color = 'blue')
    #Normalizzazione tipica delle distribuzioni binnate, prendere la distanza tra due bin edges e moltiplicarla per il numero di eventi/entrate
    ax.plot (x, N_persone * (bin_edges[1] - bin_edges[0]) * Rayleigh (x, *my_minuit.values), label = "Rayleigh Fit", color = "red")          
    ax.plot (x, N_persone * (bin_edges[1] - bin_edges[0]) * Rayleigh (x, N_passi), label = "True Rayleigh", color = "green")
    ax.set_title ('Plot persone diversamente sobrie', size = 14)
    ax.set_xlabel ('x')
    ax.set_ylabel ('Distanza percorsa')
    ax.legend ()
    ax.grid ()                                        
    plt.savefig ('Persone ubriache.png')

    plt.show ()

if __name__ == "__main__" :
    main ()



# ---- ---- ---- ---- Esame 26 settembre 2024 ---- ---- ---- ---- 

# ---- ---- Libreria/Classe ---- ----
import numpy as np
import random
from math import sqrt

# ----- ----- ----- ----- ----- ----- ----- ----- 

class additive_recurrence :

    def __init__ (self, alpha) :
        self.alpha = alpha      # a destra la varriabile, a sinistra l'attributo di self
        self.N_0 = None         # esiste l'attributo ma non esiste ancora il valore
        self.N_f = None

    def get_number (self) :
        s = (self.N_f + self.alpha)%1
        self.N_f = s
        return s
    
    def set_seed (self, seed) :
        self.N_0 = seed
        self.N_f = seed

# ----- ----- ----- ----- ----- ----- ----- ----- 

def function (x) :
    return 2 * (x**2)


# Distribuzione uniforme tra x_min e x_max con seed scelto in auto
def rand_range (x_min, x_max) :
    return x_min + random.random() * (x_max - x_min)


#Funzione per il calcolo dell'integrale (area) e scarto secondo il metodo Hit Or Miss
def MC_mod (f, x_min, x_max, y_min, y_max, N_punti, generatore) :
    x_coord = []
    y_coord = []
    for _ in range (N_punti) :
        x_coord.append (generatore.get_number ())
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


#Funzione per il calcolo dell'integrale (area ed errore)
def integral_Crude_MC (f, x_min, x_max, N_punti) :
    somma = 0.
    somma_quadrata = 0.0
    for _ in range (N_punti) :
        value = rand_range (x_min, x_max)
        somma = somma + f(value)
        somma_quadrata = somma_quadrata + ( f(value) * f(value) )
    mean = somma / N_punti
    varianza = somma_quadrata / N_punti - (mean)**2
    varianza = ( (N_punti - 1) / N_punti ) * varianza
    lunghezza = x_max - x_min
    return mean * lunghezza, sqrt (varianza / N_punti) * lunghezza


# ----- ----- Main ----- ----- 

'''
python3 main.py
'''
import numpy as np
from lib import additive_recurrence, MC_mod, function, integral_Crude_MC
import matplotlib.pyplot as plt


def main () :

    # Punto 1 e 2
    alpha = ((np.sqrt(5) - 1) / 2)
    seed = 1.
    generatore = additive_recurrence (alpha)
    generatore.set_seed (seed)

    lista_num = []
    for _ in range (0, 1000) :
        number = generatore.get_number ()
        lista_num.append (number)

    #print (lista_num[:10])

    # Punto 3
    int_val, int_incertezza = MC_mod (function, 0., 1., 0., 2., 1000, generatore)
    print ("\nValore dell'integrale: ", int_val, "+/-", int_incertezza, "\n")

    # Punto 4
    N_toys = 100
    N_points = 10
    list_points = []
    incerteza_list = []

    while (N_points < 25000) :
        integral_list = []
        for contatore_toys in range (N_toys) :
            integral_val, incertezza_val = MC_mod (function, 0., 1., 0., 2., N_points, generatore)
            integral_list.append (integral_val)
        incerteza_list.append (np.std (integral_list))         # sono indentati allo stesso modo (infatti hanno la stessa lunghezza)
        list_points.append (N_points)
        N_points = N_points * 2


    # Grafico
    fig, ax = plt.subplots ()
    ax.plot (list_points, incerteza_list, "o-", label = "Incertezza MC_mod", color = "blue")


    # Punto 5
    N_toys = 100
    N_points = 10
    list_points = []
    incerteza_list = []

    while (N_points < 25000) :
        integral_list = []
        for contatore_toys in range (N_toys) :
            integral_val, incertezza_val = integral_Crude_MC (function, 0., 1., N_points)
            integral_list.append (integral_val)
        incerteza_list.append (np.std (integral_list))         # sono indentati allo stesso modo (infatti hanno la stessa lunghezza)
        list_points.append (N_points)
        N_points = N_points * 2

    ax.plot (list_points, incerteza_list, "o-", label = "Incertezza Crude MC", color = "red")
    ax.set_xscale ("log")
    ax.set_yscale ("log")
    ax.legend ()
    ax.grid ()
    
    plt.savefig ("26settembre2024.png")
    plt.show ()
    
if __name__ == "__main__" :
    main ()




# ---- ---- ---- ---- Esame 10 ottobre 2024 ---- ---- ---- ---- 

# ---- ---- Libreria ---- ----

import random
import numpy as np
from math import ceil, sqrt

def generate_gaus_bm () :
    x1 = random.random ()
    x2 = random.random ()
    g1 = np.sqrt(-2*np.log10(x1)) * np.cos(2*np.pi*x2)
    g2 = np.sqrt(-2*np.log10(x1)) * np.sin(2*np.pi*x2)
    return g1, g2


def generate_gaus (mu, sigma) :
    g1, g2 = generate_gaus_bm ()
    g1 = g1*sigma + mu
    g2 = g2*sigma + mu
    return g1, g2

def sturges (N_eventi) :
    return ceil (1 + np.log2 (N_eventi))

# Media con lista
def media (lista) :
    mean = sum (lista)/len (lista)
    return mean

# Varianza con lista
def varianza_bessel (lista) :
    somma_quadrata = 0
    for elem in lista :
        somma_quadrata = somma_quadrata + (elem - media (lista))**2
    return somma_quadrata/(len (lista) - 1)

# Deviaz. standard con lista
def dev_std (lista) :
    sigma = (sqrt(varianza_bessel (lista)))
    return sigma

# Deviaz. standard della media con lista
def dev_std_media (lista) :
    return dev_std (lista)/sqrt (len (lista))


# ---- ---- Main ---- ----
'''
python3 ottobre24.py
'''

from lib import generate_gaus_bm, sturges, media, varianza_bessel, dev_std, dev_std_media, generate_gaus
import matplotlib.pyplot as plt
import numpy as np


def main () :

    N = 1000        # Numero di numeri pseudo-casuali da generare

    lista_casuali = []                      # inizializzo lista per numeri casuali
    for _ in range (N//2) :                    # riempimento lista
        g1, g2 = generate_gaus_bm ()
        lista_casuali.append (g1)
        lista_casuali.append (g2)

    # punto 2 esame: Istogramma
    Nbin = sturges (N)                      # per il numero di bin uso la f. di sturges
    bin_edges = np.linspace (min(lista_casuali), max(lista_casuali), Nbin)      # uso min e max
    
    fig, ax = plt.subplots (nrows = 1, ncols = 1)
    ax.hist (lista_casuali, bins = bin_edges, color = 'orange')
    ax.set_title ('Istogramma', size = 14)
    ax.set_xlabel ('x')
    ax.set_ylabel ('Conteggi')
    ax.grid ()
    
    plt.savefig ('Ottobre 2024.png')
    
    # punto 3 esame: Statistiche
    print ("Media della distribuzione: ", media (lista_casuali))
    print ("Varianza della distribuzione: ", varianza_bessel (lista_casuali))

    # punto 4 esame: mostrare che dev_standard non cambia e dev_standard della media si
    n = 1
    lista_casuali_stat = []
    sigma_list = []
    sigma_mean_list = []
    
    while (n <= 1000) :

        g1, g2 = generate_gaus_bm ()
        lista_casuali_stat.append (g1)
        lista_casuali_stat.append (g2)

        sigma_list.append (dev_std(lista_casuali_stat))
        sigma_mean_list.append (dev_std_media(lista_casuali_stat))
        n = n + 2
        
    x_axis = np.linspace (2, len(lista_casuali_stat), 500)

    # Creazione di una figura con due subplot affiancati
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))  # 1 riga, 2 colonne
    
    # Primo grafico: mostra l'andamento della deviazione standard
    axes[0].plot(x_axis, sigma_list, label="sigma") 
    axes[0].legend()
    axes[0].grid()
    axes[0].set_title("Andamento sigma")

    # Secondo grafico: mostra l'andamento della deviazione standard della media
    axes[1].plot(x_axis, sigma_mean_list, label="Sigma mean")
    axes[1].legend()
    axes[1].grid()
    axes[1].set_title("Andamento sigma media")

    plt.tight_layout()  # Questo aggiusta automaticamente spazi tra i grafici
    plt.savefig("Confronto media con sigma_media.png")

    # punto 5

    mu = 5.
    sigma = 2.
    
    list_gaus_histo = []

    for _ in range (N//2) :                    # riempimento lista
        g1, g2 = generate_gaus (mu, sigma)
        list_gaus_histo.append (g1)
        list_gaus_histo.append (g2)

    # punto 2 esame: Istogramma
    Nbin = sturges (N)                                                              # per il numero di bin uso la f. di sturges
    bin_edges = np.linspace (min(list_gaus_histo), max(list_gaus_histo), Nbin)      # uso min e max
    
    fig, ax = plt.subplots (nrows = 1, ncols = 1)
    ax.hist (list_gaus_histo, bins = bin_edges, color = 'orange')
    ax.set_title ('Istogramma 2', size = 14)
    ax.set_xlabel ('x')
    ax.set_ylabel ('Conteggi')
    ax.grid ()
    
    plt.savefig ('Ottobre 2024 gaus.png')

    plt.show()

if __name__ == "__main__" :
    main ()