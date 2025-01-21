'''
python3 luglio2024.py
'''
import numpy as np
import matplotlib.pyplot as plt
from iminuit import Minuit
from iminuit.cost import ExtendedBinnedNLL
from scipy.stats import rayleigh
from lib import random_walk, calcola_distanza, media, varianza_bessel, skewness, kurtosis, sturges, funzione_fit, Rayleigh

# Lib

import numpy as np
import random
from math import ceil
from scipy.stats import rayleigh

def sturges (N_eventi) :
    return int (ceil (1 + np.log2 (N_eventi))) 

# ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ----

# Distribuzione uniforme tra x_min e x_max con seed scelto in auto
def rand_range (x_min, x_max) :
    return x_min + random.random() * (x_max - x_min)

# ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ----

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

# ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ----

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

# ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ----

def calcola_distanza (x_0, x_n, y_0, y_n) :
    return np.sqrt( ((x_n - x_0)**2) + ((y_n - y_0)**2) )

# ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ----

def funzione_fit (bin_edges, N) :
    return rayleigh.cdf (bin_edges, loc = 0, scale = np.sqrt (N/2))

# ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ----

def Rayleigh (r, N) :
    return ((2*r)/N) * np.exp(-(r**2)/N)


# -------------- Statistiche --------------

# Media con array
def media (sample) :
    mean = np.sum(sample)/len(sample)
    return mean
    
# ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ----

# Varianza con array
def varianza (sample) :
    somma_quadrata = 0
    somma_quadrata = np.sum( (sample - media (sample))**2 )
    var = somma_quadrata/(len (sample))
    return var

# ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ----

# Varianza con corr. di Bessel con array
def varianza_bessel (sample) :
    somma_quadrata = 0
    somma_quadrata = np.sum( (sample - media(sample))**2 )
    var = somma_quadrata/(len(sample) - 1)
    return var

# ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ----

# Deviaz. standard con array
def dev_std (sample) :
    sigma = np.sqrt (varianza(sample))
    return sigma

# ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ----

# Deviaz. standard della media con array
def dev_std_media (sample) :
    return dev_std(sample) / (np.sqrt( len(sample) ))

# ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ----

# Skewness con array
def skewness (sample) :
    mean = media (sample)  # Calcola la media con la tua funzione
    sigma = dev_std (sample)  # Calcola la deviazione standard con la tua funzione
    n = len(sample)
    skew = np.sum((sample - mean)**3) / (n * sigma**3)
    return skew

# ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ----

# Curtosi con array
def kurtosis (sample) :
    mean = media (sample)  # Calcola la media con la tua funzione
    variance = varianza (sample)  # Calcola la varianza con la tua funzione
    n = len(sample)
    kurt = np.sum((sample - mean)**4) / (n * variance**2) - 3
    return kurt


# Main

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

    distanza = calcola_distanza (0., coord_x_array[10] , 0., coord_y_array[10])
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
