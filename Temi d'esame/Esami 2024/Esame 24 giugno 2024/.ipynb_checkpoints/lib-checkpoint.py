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