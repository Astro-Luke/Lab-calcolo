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